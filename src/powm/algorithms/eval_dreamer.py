import random
from collections import defaultdict
from functools import partial as bind

import dreamerv3
import jax
import numpy as np
import torch
from dreamerv3 import embodied

from powm.algorithms.train_dreamer import make_env, make_logger, make_replay
from powm.mine.mine import MutualInformationNeuralEstimator


def epsilon_greedy_policy(agent, eps):
    random_agent = embodied.RandomAgent(agent.obs_space, agent.act_space)

    def policy(*args, **kwargs):
        acts, out, (lat, act) = agent.policy(*args, mode="eval", **kwargs)
        if random.random() < eps:
            acts, _, _ = random_agent.policy(*args, mode="eval", **kwargs)
        act = jax.device_put(acts, agent.policy_sharded)
        return acts, out, (lat, act)

    return policy


def collect_beliefs(logger, agent, config, driver, n_samples, eps):
    pred_beliefs = [[] for _ in range(driver.length)]
    true_beliefs = [[] for _ in range(driver.length)]
    driver.reset(agent.init_policy)
    policy = epsilon_greedy_policy(agent, eps)
    episodes = defaultdict(embodied.Agg)
    ckpt_stats = embodied.Agg()

    def log_step(tran, worker):
        nonlocal pred_beliefs, true_beliefs
        ep_stats = episodes[worker]
        ep_stats.add("score", tran["reward"], agg="sum")
        ep_stats.add("length", 1, agg="sum")
        ep_stats.add("rewards", tran["reward"], agg="stack")
        if driver.parallel:
            [pipe.send(("info",)) for pipe in driver.pipes]
            infos = [driver._receive(pipe) for pipe in driver.pipes]
        else:
            infos = [env.info for env in driver.envs]
        true_beliefs[worker].append(infos[worker]["belief"])
        deter = tran["deter"]
        stoch = tran["stoch"]
        n_classes = config.dyn.rssm.classes
        stoch_one_hot = np.zeros((len(stoch), n_classes))
        stoch_one_hot[np.arange(len(stoch)), stoch] = 1
        pred_beliefs[worker].append(np.concatenate([deter, stoch_one_hot.reshape(-1)]))
        if tran["is_last"]:
            result = ep_stats.result()
            ckpt_stats.add("score", result["score"], agg="avg")
            ckpt_stats.add("length", result["length"], agg="avg")

            # Calculate discounted returns
            rewards = result["rewards"]
            discount_factor = 1 - 1 / config.horizon
            discounts = discount_factor ** np.arange(len(rewards))
            discounted_return = np.sum(rewards * discounts)
            ckpt_stats.add("return", discounted_return, agg="avg")

    driver.callbacks = []
    driver.on_step(log_step)
    driver(policy, steps=n_samples)

    if logger is not None:
        logger.add(ckpt_stats.result(), prefix=f"eval_eps_{eps}")
        logger.write()

    pred_beliefs = np.concatenate(pred_beliefs, axis=0)
    true_beliefs = np.concatenate(true_beliefs, axis=0)
    return torch.tensor(pred_beliefs, dtype=torch.float32), torch.tensor(
        true_beliefs, dtype=torch.float32
    )


def main(argv=None):
    parsed, other = embodied.Flags(logdir="").parse_known(argv)
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    results = {}
    config = embodied.Config.load(str(logdir / "config.yaml"))

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create the environment once
    env = make_env(config, set_jax_flags=False)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    logger = make_logger(config)
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        batch_length_eval=config.batch_length_eval,
        replay_length=config.replay_length,
        replay_length_eval=config.replay_length_eval,
        replay_context=config.replay_context,
    )

    fns = [
        bind(make_env, config, env_id=i, estimate_belief=True, set_jax_flags=True)
        for i in range(args.num_envs)
    ]
    driver = embodied.Driver(fns, args.driver_parallel)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.step = embodied.Counter()
    checkpoint.replay = make_replay(config)
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        checkpoint.load(ckpt_path)
        logger.step = checkpoint.step

        for eps in [0.0, 0.5, 1.0]:
            train_pred_beliefs, train_true_beliefs = collect_beliefs(
                logger, agent, config, driver, 10000, eps
            )

            mine = MutualInformationNeuralEstimator(
                hs_sizes=train_pred_beliefs.size(-1),
                belief_sizes=[train_true_beliefs.size(-1)],
                hidden_size=256,
                num_layers=2,
                alpha=0.01,
                representation_sizes=[16],
                belief_part=None,
                device="cuda",
            )
            mine.optimize(
                train_pred_beliefs,
                (train_true_beliefs,),
                num_epochs=100,
                logger=lambda x: None,
                learning_rate=1e-3,
                batch_size=1024,
                lambd=0.0,
                valid_size=0.2,
            )
            driver.reset(agent.init_policy)
            test_pred_beliefs, test_true_beliefs = collect_beliefs(
                None, agent, config, driver, 1000, eps
            )
            train_mi = mine.estimate(train_pred_beliefs, (train_true_beliefs,))
            test_mi = mine.estimate(test_pred_beliefs, (test_true_beliefs,))
            logger.add(
                {"train_mi": train_mi, "test_mi": test_mi}, prefix=f"eval_eps_{eps}"
            )
            logger.write()

    env.close()
    return results


# Example usage
if __name__ == "__main__":
    main()
