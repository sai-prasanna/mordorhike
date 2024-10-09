import random
from collections import defaultdict
from functools import partial as bind

import dreamerv3
import jax
import jax.numpy as jnp
import numpy as np
import torch
from dreamerv3 import embodied
from dreamerv3.jaxagent import fetch_async

from powm.algorithms.train_dreamer import make_env, make_logger
from powm.mine.mine import MutualInformationNeuralEstimator


def epsilon_greedy_policy(agent, eps, policy_mode="train"):
    random_agent = embodied.RandomAgent(agent.obs_space, agent.act_space)

    def policy(*args, **kwargs):
        acts, out, (lat, act) = agent.policy(*args, mode=policy_mode, **kwargs)
        if random.random() < eps:
            acts, _, _ = random_agent.policy(*args, mode=policy_mode, **kwargs)
        act = jax.device_put(acts, agent.policy_sharded)
        return acts, out, (lat, act)

    return policy


def collect_beliefs(logger, agent, config, driver, n_samples, eps, policy_mode):
    pred_beliefs = [[] for _ in range(driver.length)]
    true_beliefs = [[] for _ in range(driver.length)]
    driver.reset(agent.init_policy)
    policy = epsilon_greedy_policy(agent, eps, policy_mode)
    episodes = defaultdict(embodied.Agg)
    ckpt_stats = embodied.Agg()
    replay = embodied.replay.Replay(
        length=config.batch_length_eval,
        capacity=config.replay.size,
        online=True,
    )

    def log_step(tran, worker):
        nonlocal pred_beliefs, true_beliefs
        ep_stats = episodes[worker]
        ep_stats.add("score", tran["reward"], agg="sum")
        ep_stats.add("length", 1, agg="sum")
        ep_stats.add("rewards", tran["reward"], agg="stack")
        if worker == 0:
            for key in config.run.log_keys_video:
                if key in tran:
                    ep_stats.add(f"policy_{key}", tran[key], agg="stack")
        if driver.parallel:
            [pipe.send(("info",)) for pipe in driver.pipes]
            infos = [driver._receive(pipe) for pipe in driver.pipes]
        else:
            infos = [env.info for env in driver.envs]
        true_beliefs[worker].append(infos[worker]["belief"])
        deter = tran["deter"]
        stoch = tran["stoch"]
        n_classes = config.dyn.rssm.classes
        n_particles = config.n_particles
        stoch_one_hot = np.zeros((*stoch.shape, n_classes))
        # Add the one-hot encoding. stoch is particle x entries and stoch_onehot is particle x entries x n_classes
        for i in range(n_particles):
            stoch_one_hot[i, np.arange(len(stoch[i])), stoch[i]] = 1

        # Flatten the last two dimensions of stoch_one_hot
        stoch_one_hot_flattened = stoch_one_hot.reshape(*stoch_one_hot.shape[:-2], -1)

        pred_beliefs[worker].append(
            np.concatenate([deter, stoch_one_hot_flattened], axis=-1)
        )
        if tran["is_last"]:
            result = ep_stats.result()
            ckpt_stats.add("score", result.pop("score"), agg="avg")
            ckpt_stats.add("length", result.pop("length"), agg="avg")

            # Calculate discounted returns
            rewards = result.pop("rewards")
            discount_factor = 1 - 1 / config.horizon
            discounts = discount_factor ** np.arange(len(rewards))
            discounted_return = np.sum(rewards * discounts)
            ckpt_stats.add("return", discounted_return, agg="avg")
            ckpt_stats.add(result)

    driver.callbacks = []
    driver.on_step(log_step)
    driver.on_step(replay.add)

    driver(policy, steps=n_samples)
    dataset_eval = iter(
        agent.dataset(bind(replay.dataset, config.batch_size, config.batch_length_eval))
    )
    errors = None
    total_weights = None
    for batch in dataset_eval:
        if not fetch_async(jnp.any(batch["is_online"])):
            break
        batch["is_terminal"] = batch["is_terminal"] & batch["is_online"]
        batch_errors = agent.report_wm_prediction_error(batch)
        if errors is None:
            errors = {
                k: np.zeros_like(batch_errors[k]["mean"]) for k in batch_errors.keys()
            }
            total_weights = {
                k: np.zeros_like(batch_errors[k]["weight"]) for k in batch_errors.keys()
            }
        for k, v in batch_errors.items():
            errors[k] += v["mean"] * total_weights[k]
            total_weights[k] += v["weight"]
    # After processing all batches
    weighted_mean_errors = {
        k: errors[k] / np.maximum(total_weights[k], 1) for k in errors.keys()
    }
    if logger is not None:
        logger_prefix = f"eval_eps_{eps}"
        for k, v in weighted_mean_errors.items():
            record = {
                f"wm_pred_error_step_{step+1}_{k}": v[step] for step in range(len(v))
            }
            logger.add(record, prefix=logger_prefix)
        logger.add(ckpt_stats.result(), prefix=logger_prefix)
        logger.write()

    pred_beliefs = np.concatenate(pred_beliefs, axis=0)
    true_beliefs = np.concatenate(true_beliefs, axis=0)
    return torch.tensor(pred_beliefs, dtype=torch.float32), torch.tensor(
        true_beliefs, dtype=torch.float32
    )


def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        mine_epochs=100,
        mine_hidden_size=256,
        mine_layers=2,
        mine_alpha=0.01,
        mine_deep_set_size=16,
        metric_dir="eval",
        policy_mode="train",
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    # config = config.update({"jax.jit": False, "jax.transfer_guard": False})

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create the environment once
    env = make_env(config)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    logger = make_logger(config, metric_dir=parsed.metric_dir)

    fns = [bind(make_env, config, env_id=i, estimate_belief=True) for i in range(1)]
    driver = embodied.Driver(fns, config.run.driver_parallel)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.step = embodied.Counter()
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        checkpoint.load(ckpt_path, keys=["agent", "step"])
        logger.step = checkpoint.step

        for eps in [0.0, 0.5, 1.0]:
            train_pred_beliefs, train_true_beliefs = collect_beliefs(
                logger,
                agent,
                config,
                driver,
                10 * config.batch_size * config.batch_length_eval,
                eps,
                policy_mode=parsed.policy_mode,
            )

            mine = MutualInformationNeuralEstimator(
                hs_size=train_pred_beliefs.size(-1),
                belief_sizes=[train_true_beliefs.size(-1)],
                hidden_size=parsed.mine_hidden_size,
                num_layers=parsed.mine_layers,
                alpha=parsed.mine_alpha,
                hidden_deepset_encode_size=parsed.mine_deep_set_size,
                belief_deepset_encode_sizes=[parsed.mine_deep_set_size],
                belief_part=None,
                device="cuda",
            )

            best_train_mi = 0
            best_valid_mi = 0
            best_epoch = 0

            def mi_logger(record):
                nonlocal best_train_mi, best_valid_mi, best_epoch
                best_train_mi = record["mine_optim/best_train_mi"]
                best_valid_mi = record["mine_optim/best_valid_mi"]
                best_epoch = record["mine_optim/best_epoch"]

            mine.optimize(
                train_pred_beliefs,
                (train_true_beliefs,),
                num_epochs=100,
                logger=mi_logger,
                learning_rate=1e-3,
                batch_size=1024,
                lambd=0.0,
                valid_size=0.2,
            )
            driver.reset(agent.init_policy)
            test_pred_beliefs, test_true_beliefs = collect_beliefs(
                None,
                agent,
                config,
                driver,
                2 * config.batch_size * config.batch_length_eval,
                eps,
                policy_mode=parsed.policy_mode,
            )
            test_mi = mine.estimate(test_pred_beliefs, (test_true_beliefs,))
            logger.add(
                {
                    "train_mi": best_train_mi,
                    "valid_mi": best_valid_mi,
                    "train_mi_epoch": best_epoch,
                    "test_mi": test_mi,
                },
                prefix=f"eval_eps_{eps}",
            )
            logger.write()
    logger.close()
    driver.close()


# Example usage
if __name__ == "__main__":
    main()
