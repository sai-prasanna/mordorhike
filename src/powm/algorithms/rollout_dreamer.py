import random
from collections import defaultdict
from functools import partial as bind
import re

import dreamerv3
import numpy as np
import torch
from dreamerv3 import embodied
from dreamerv3 import jaxutils

from powm.algorithms.train_dreamer import make_env, make_logger


def collect_rollouts(logger, agent, config, driver: embodied.Driver, num_episodes):
    episodes_data = []
    current_episodes = {i: defaultdict(list) for i in range(driver.length)}
    scores = []
    lengths = []
    returns = []
    
    driver.reset(agent.init_policy)
    episodes = defaultdict(embodied.Agg)

    def log_step(tran, worker):
        ep_stats = episodes[worker]
        ep_stats.add("score", tran["reward"], agg="sum")
        ep_stats.add("length", 1, agg="sum")
        ep_stats.add("rewards", tran["reward"], agg="stack")

        # Get environment info
        if driver.parallel:
            [pipe.send(("info",)) for pipe in driver.pipes]
            infos = [driver._receive(pipe) for pipe in driver.pipes]
        else:
            infos = [env.info for env in driver.envs]
        info = infos[worker]
        current_episode = current_episodes[worker]
        # Collect episode data
        # Assume we are encoding and decoding same observation spaces
        # which is standard for dreamerv3 or any sequential VAEs
        matching_keys = [k for k in tran.keys() if re.match(config.dec.spaces, k)]
        for k in matching_keys:
            key = "obs" if len(matching_keys) == 1 else f"obs_{k}"
            current_episode[key].append(tran[k])
        for k in driver.act_space.keys():
            current_episode[k].append(tran[k])
        current_episode["reward"].append(tran["reward"])
        current_episode["state"].append(info["state"])
        current_episode["belief"].append(info["belief"])
        
        # Store raw deter and stoch temporarily
        current_episode["_deter"].append(tran["deter"])
        current_episode["_stoch"].append(tran["stoch"])

        if tran["is_last"]:
            result = ep_stats.result()
            scores.append(result["score"])
            lengths.append(result["length"])
            
            # Calculate discounted return
            rewards = result["rewards"]
            discount_factor = 1 - 1 / config.horizon
            discounts = discount_factor ** np.arange(len(rewards))
            discounted_return = np.sum(rewards * discounts)
            returns.append(discounted_return)
            
            # Pass raw deter and stoch to prediction
            episode_latents = {
                'deter': np.array(current_episode['_deter'])[np.newaxis, ...],
                'stoch': np.array(current_episode['_stoch'])[np.newaxis, ...]
            }
            episode_actions = {k: np.array(v)[np.newaxis, ...] for k, v in current_episode.items() 
                             if k in driver.act_space.keys() and k != 'reset'}
            
            # Compute predictions
            predictions = agent.multi_step_prediction(
                episode_latents,
                episode_actions
            )
            num_keys = len(predictions)
            for k, v in predictions.items():
                key = "obs_hat" if num_keys == 1 else f"obs_hat_{k}"
                # Remove batch as it's 1
                v = v[0] # Timestep * n_steps * particles * pred_dim
                # swap particles and n_steps
                v = v.swapaxes(1, 2) # Timestep * particles * n_steps * pred_dim
                current_episode[key] = v

            # Create one-hot latents for storage
            deter = np.array(current_episode['_deter'])
            stoch = np.array(current_episode['_stoch'])
            # Convert categorical stoch to one-hot
            stoch_one_hot = np.eye(config.dyn.rssm.classes)[stoch]
            # Flatten the one-hot dimension
            stoch_one_hot_flat = stoch_one_hot.reshape(*stoch_one_hot.shape[:-2], -1)
            # Concatenate with deter to get final latent
            latents = np.concatenate([deter, stoch_one_hot_flat], axis=-1)
            del current_episode['_deter']
            del current_episode['_stoch']
            episodes_data.append({
                **{k: np.array(v) for k, v in current_episode.items()},
                'latents': latents,
            })
            current_episode.clear()


    driver.callbacks = []
    driver.on_step(log_step)
    driver(agent.policy, episodes=num_episodes)

    # Log statistics
    if logger is not None:
        logger.add({
            'score_mean': np.mean(scores),
            'score_std': np.std(scores),
            'length_mean': np.mean(lengths),
            'length_std': np.std(lengths),
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
        })
        logger.write()

    return episodes_data

def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
        policy_mode="train",
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create the environment once
    env = make_env(config, index=0)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    logger = make_logger(config, metric_dir=parsed.metric_dir)
    env.close()

    fns = [bind(make_env, config, index=i, estimate_belief=True) for i in range(config.run.num_envs)]
    driver = embodied.Driver(fns, config.run.driver_parallel)
    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.step = embodied.Counter()
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        checkpoint.load(ckpt_path, keys=["agent", "step"])
        logger.step = checkpoint.step
        # Collect training data
        episodes = collect_rollouts(
            logger,
            agent,
            config,
            driver,
            110,  # num training episodes
        )
        # Save episode data
        ckpt_number = int(logger.step)
        np.savez(
            f"{parsed.logdir}/episodes_{ckpt_number}.npz",
            episodes=episodes,
        )

    logger.close()
    driver.close()


# Example usage
if __name__ == "__main__":
    # main()
    main()