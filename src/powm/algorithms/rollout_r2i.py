import random
from collections import defaultdict
import re

import numpy as np
import recall2imagine
from recall2imagine import embodied

from powm.algorithms.train_r2i import make_logger, make_envs


def collect_rollouts(agent, config, env, num_episodes):
    episodes_data = []
    worker_episodes = defaultdict(lambda: defaultdict(list))
    scores = []
    lengths = []
    returns = []
    
    driver = embodied.Driver(env)
    driver.reset()

    def log_step(tran, worker):
        # Get environment info
        info = env._envs[worker].info
        current_episode = worker_episodes[worker]
        # Collect episode data
        # Assume we are encoding and decoding same observation spaces
        
        matching_keys = [k for k in tran.keys() 
                         if re.match(config.decoder.mlp_keys, k) or re.match(config.decoder.cnn_keys, k)]
        for k in matching_keys:
            key = "obs" if len(matching_keys) == 1 else f"obs_{k}"
            current_episode[key].append(tran[k])
        for k in env.act_space.keys():
            current_episode[k].append(tran[k])
        current_episode["reward"].append(tran["reward"])
        current_episode["state"].append(info["state"])
        current_episode["belief"].append(info["belief"])
        
        #Store raw deter and stoch temporarily
        latent_keys = ["deter", "stoch", "hidden", "logit"]
        for k in latent_keys:
            if k in tran.keys():
                current_episode[k].append(tran[k])
        
        if tran["is_last"]:
            # Calculate episode statistics
            score = sum(current_episode["reward"])
            length = len(current_episode["reward"])
            scores.append(score)
            lengths.append(length)
            
            # Calculate discounted return
            rewards = np.array(current_episode["reward"])
            discount_factor = 1 - 1 / config.horizon
            discounts = discount_factor ** np.arange(len(rewards))
            discounted_return = np.sum(rewards * discounts)
            returns.append(discounted_return)
            
            # Get latent trajectories
            episode_latents = {}
            for k in ["deter", "stoch", "hidden", "logit"]:
                episode_latents[k] = np.array(current_episode[k])[np.newaxis, ...]
            episode_actions = {k: np.array(v)[np.newaxis, ...] for k, v in current_episode.items() 
                             if k in env.act_space.keys() and k != 'reset'}
            
            # Compute predictions
            predictions = agent.multi_step_prediction(
                episode_latents,
                episode_actions
            )
            num_keys = len(predictions)
            for k, v in predictions.items():
                key = f"pred_{k}" if num_keys == 1 else f"pred_{k}"
                current_episode[key].append(v)
            # Create one-hot latents for storage
            deter = np.array(current_episode['deter'])
            stoch = np.array(current_episode['stoch'])
            stoch = stoch.reshape(*stoch.shape[:-2], -1)
            hidden = np.array(current_episode['hidden'])
            hidden = np.stack([hidden.real, hidden.imag], axis=-1)
            hidden = hidden.reshape(*hidden.shape[:-3], -1)
            # Concatenate with deter to get final latent
            latents = np.concatenate([deter, stoch, hidden], axis=-1)
            for k in latent_keys:
                del current_episode[k]
            
            # Store episode data with predictions
            episodes_data.append({
                **{k: np.array(v) for k, v in current_episode.items()},
                'latents': latents,
                "success": tran["is_terminal"]
            })
            current_episode.clear()

    driver.on_step(log_step)
    driver(agent.policy, episodes=num_episodes, record_state=True)
    return episodes_data


def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    config.update({"envs.amount": 4})

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)

    # Create environment and agent
    env = make_envs(config, estimate_belief=True)
    step = embodied.Counter()
    agent = recall2imagine.Agent(env.obs_space, env.act_space, step, config)
    logger = make_logger(logdir, step, config, metric_dir=parsed.metric_dir)

    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        checkpoint = embodied.Checkpoint(ckpt_path)
        checkpoint.step = step
        checkpoint.agent = agent
        checkpoint.load(ckpt_path, keys=["agent", "step"])
        
        
        # Collect training data
        episodes = collect_rollouts(
            agent,
            config,
            env,
            110,  # num training episodes
        )
        
        # Save episode data
        ckpt_number = int(step)
         # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{ckpt_number}.npz",
            episodes=episodes
        )

    env.close()


if __name__ == "__main__":
    main()
