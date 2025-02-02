import random
import re
from collections import defaultdict

import numpy as np
import recall2imagine
from recall2imagine import embodied

from powm.algorithms.train_r2i import make_envs, make_logger
from powm.utils import set_seed


def collect_rollouts(
    checkpoint_path: str | None,
    config,
    env,
    num_episodes: int,
    epsilon: float = 0.0,
    collect_only_rewards: bool = False
):
    """Collect rollouts from a R2I agent.
    
    Args:
        checkpoint_path: Path to checkpoint to load, or None for random agent
        config: Configuration object
        driver: Driver for environment interaction
        num_episodes: Number of episodes to collect
        epsilon: Exploration rate
        collect_only_rewards: If True, only collect rewards
    """
    # Create agent
    agent = recall2imagine.Agent(env.obs_space, env.act_space, embodied.Counter(), config)
    control_agent = recall2imagine.Agent(env.obs_space, env.act_space, embodied.Counter(), config)

    checkpoint = embodied.Checkpoint()
    checkpoint.step = embodied.Counter()
    checkpoint.agent = agent
    checkpoint.load(checkpoint_path, keys=['agent', 'step'])

    episodes_data = []
    worker_episodes = defaultdict(lambda: defaultdict(list))
    scores = []
    lengths = []
    returns = []
    
    driver = embodied.Driver(env)
    driver.reset()

    def log_step(tran, worker):
        current_episode = worker_episodes[worker]
        
        # Always collect rewards for score calculation
        current_episode["reward"].append(tran["reward"])
        
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
            
            if collect_only_rewards:
                episodes_data.append({
                    'reward': np.array(current_episode["reward"]),
                })
                current_episode.clear()
                return
                
            # Get environment info and collect full data
            info = env._envs[worker].info
            
            # Collect episode data
            matching_keys = [k for k in tran.keys() 
                           if re.match(config.decoder.mlp_keys, k) or re.match(config.decoder.cnn_keys, k)]
            for k in matching_keys:
                key = "obs" if len(matching_keys) == 1 else f"obs_{k}"
                current_episode[key].append(tran[k])
            for k in env.act_space.keys():
                current_episode[k].append(tran[k])
            current_episode["state"].append(info["state"])
            current_episode["belief"].append(info["belief"])
            
            # Store raw deter and stoch temporarily
            latent_keys = ["deter", "stoch", "hidden", "logit", "control_deter", "control_stoch", "control_hidden"]
            for k in latent_keys:
                if k in tran.keys():
                    current_episode[k].append(tran[k])
            
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
                key = f"obs_hat" if num_keys == 1 else f"pred_{k}"
                v = v.swapaxes(0, 1)  # Swap timestep and batch dimensions
                current_episode[key] = v
                
            # Create latents for storage
            def create_latents(deter, stoch, hidden):
                deter = np.array(deter)
                stoch = np.array(stoch)
                stoch = stoch.reshape(*stoch.shape[:-2], -1)
                hidden = np.array(hidden)
                hidden = np.stack([hidden.real, hidden.imag], axis=-1)
                hidden = hidden.reshape(*hidden.shape[:-3], -1)
                return np.concatenate([deter, stoch, hidden], axis=-1)
            
            latents = create_latents(current_episode['deter'], current_episode['stoch'], current_episode['hidden'])
            control_latents = create_latents(current_episode['control_deter'], current_episode['control_stoch'], current_episode['control_hidden'])
            
            for k in latent_keys:
                del current_episode[k]
            
            # Store episode data with predictions
            current_episode["action"] = np.argmax(current_episode["action"], axis=-1)
            episodes_data.append({
                **{k: np.array(v) for k, v in current_episode.items()},
                'latent': latents[:, np.newaxis, ...],
                'control_latent': control_latents[:, np.newaxis, ...],
                "success": tran["is_terminal"]
            })
            current_episode.clear()

    control_state = None
    def policy(obs, state, **kwargs):
        nonlocal control_state
        if state and 'control_deter' in state[0][0].keys():
            del state[0][0]['control_deter']
            del state[0][0]['control_stoch']
            del state[0][0]['control_hidden']
        acts, state = agent.policy(obs, state, **kwargs)
        _, control_state = control_agent.policy(obs, control_state, **kwargs)
        
        if epsilon != 0.0:
            eps_acts = {}
            n_envs = len(driver._env)
            eps_mask = np.random.random(size=n_envs) < epsilon
            for k in acts.keys():
                if k.startswith('log_'):
                    continue
                act_space = driver._env.act_space[k]
                random_acts = np.stack([act_space.sample() for _ in range(n_envs)])
                eps_acts[k] = np.where(eps_mask[:, None], random_acts, acts[k])
            (prev_latent, _), task_state, expl_state = state
            state = (prev_latent, eps_acts['action']), task_state, expl_state
            
        control_state = (control_state[0][0], state[0][1]), control_state[1], control_state[2]
        state[0][0]['control_deter'] = control_state[0][0]['deter']
        state[0][0]['control_stoch'] = control_state[0][0]['stoch']
        state[0][0]['control_hidden'] = control_state[0][0]['hidden']
        return acts, state
    driver.on_step(log_step)
    driver(policy, episodes=num_episodes, record_state=True)
    return episodes_data


def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        collect_n_episodes=300,
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    config.update({"envs.amount": 4})

    # Set seeds
    set_seed(config.seed)

    # Create environment and agent
    env = make_envs(config, estimate_belief=True)
    step = embodied.Counter()
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        # Collect training data
        episodes = collect_rollouts(
            ckpt_path,
            config,
            env,
            parsed.collect_n_episodes,
        )
        noisy_episodes = collect_rollouts(
            ckpt_path,
            config,
            env,
            parsed.collect_n_episodes,
            epsilon=0.25
        )
        # Save episode data
        ckpt_number = int(step)
        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{ckpt_number}.npz",
            episodes=episodes,
            noisy_episodes=noisy_episodes
        )

    env.close()


if __name__ == "__main__":
    main()