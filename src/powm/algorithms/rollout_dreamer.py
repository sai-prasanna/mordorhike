import random
import re
from collections import defaultdict
from functools import partial as bind

import dreamerv3
import jax
import numpy as np
from dreamerv3 import embodied

from powm import envs
from powm.algorithms.train_dreamer import make_env, make_logger
from powm.utils import set_seed


def collect_rollouts(
    checkpoint_path: str | None,
    config,
    driver: embodied.Driver,
    num_episodes: int,
    epsilon: float = 0.0,
    collect_only_rewards: bool = False,
    waypoints: list = None,
):
    """Collect rollouts from a Dreamer agent.
    
    Args:
        checkpoint_path: Path to checkpoint to load, or None for random agent
        config: Configuration object
        driver: Driver for environment interaction
        num_episodes: Number of episodes to collect
        epsilon: Exploration rate
        collect_only_rewards: If True, only collect rewards
        waypoints: If provided, list of waypoints to follow
    """
    # Create agent
    env = make_env(config, index=0)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    control_agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()

    if checkpoint_path is not None:
        checkpoint = embodied.Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(checkpoint_path, keys=['agent'])

    episodes_data = []
    current_episodes = {i: defaultdict(list) for i in range(driver.length)}
    scores = []
    lengths = []
    returns = []
    
    driver.reset(agent.init_policy)
    episodes = defaultdict(embodied.Agg)

    # Initialize waypoint tracking
    visited_waypoints = [None] * driver.length
    following_waypoints = [True] * driver.length if waypoints is not None else None
    waypoint_env = make_env(config, index=0) if waypoints is not None else None
    final_waypoint_steps = [None] * driver.length  # Track when final waypoint was reached

    def log_step(tran, worker):
        nonlocal waypoints
        ep_stats = episodes[worker]
        ep_stats.add("score", tran["reward"], agg="sum")
        ep_stats.add("length", 1, agg="sum")
        ep_stats.add("rewards", tran["reward"], agg="stack")
        ep_stats.add("log_image", tran["log_image"], agg="stack")

        if collect_only_rewards:
            current_episode = current_episodes[worker]
            current_episode["reward"].append(tran["reward"])
            if tran["is_last"]:
                result = ep_stats.result()
                scores.append(result["score"])
                lengths.append(result["length"])
                rewards = result["rewards"]
                discount_factor = 1 - 1 / config.horizon
                discounts = discount_factor ** np.arange(len(rewards))
                discounted_return = np.sum(rewards * discounts)
                returns.append(discounted_return)
                episodes_data.append({
                    'reward': np.array(current_episode['reward']),
                })
                current_episode.clear()
            return

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
        current_episode["_control_deter"].append(tran["control_deter"])
        current_episode["_control_stoch"].append(tran["control_stoch"])

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

            def convert_to_latent(deter, stoch):
                stoch_one_hot = np.eye(config.dyn.rssm.classes)[stoch]
                # Flatten the one-hot dimension
                stoch_one_hot_flat = stoch_one_hot.reshape(*stoch_one_hot.shape[:-2], -1)
                
                # Concatenate with deter to get final latent
                latents = np.concatenate([deter, stoch_one_hot_flat], axis=-1)
                return latents
            # Create one-hot latents for storage
            latents = convert_to_latent(
                np.array(current_episode['_deter']),
                np.array(current_episode['_stoch'])
            )
            control_latents = convert_to_latent(
                np.array(current_episode['_control_deter']),
                np.array(current_episode['_control_stoch'])
            )
            # Convert categorical stoch to one-hot
            del current_episode['_deter']
            del current_episode['_stoch']
            del current_episode['_control_deter']
            del current_episode['_control_stoch']
            
            episode_data = {
                **{k: np.array(v) for k, v in current_episode.items()},
                'latent': latents,
                'control_latent': control_latents,
                "success": tran["is_terminal"],
            }
            if waypoints is not None:
                episode_data["final_waypoint_step"] = np.array(final_waypoint_steps[worker])
                episode_data["waypoints"] = waypoints
                # Reset waypoint tracking for this worker
                visited_waypoints[worker] = None
                following_waypoints[worker] = True
                final_waypoint_steps[worker] = None
            episodes_data.append(episode_data)
            current_episode.clear()

    control_carry = None
    waypoint_env = None
    
    
    def policy(obs, carry, **kwargs):
        nonlocal control_carry, visited_waypoints, following_waypoints, waypoint_env, final_waypoint_steps
        
        if control_carry is None:
            control_carry = carry
            
        acts, outs, carry = agent.policy(obs, carry, **kwargs)
        _, control_outs, control_carry = control_agent.policy(obs, control_carry, **kwargs)
        
        # Handle waypoint navigation
        if waypoints is not None:
            for i in range(driver.length):
                if following_waypoints[i]:
                    waypoint_action, visited_waypoints[i] = driver.envs[i]._env.unwrapped.get_waypoint_action(waypoints, visited_waypoints[i])
                    if waypoint_action is not None:
                        acts['action'] = acts['action'].copy()
                        acts['action'][i] = waypoint_action
                    else:
                        following_waypoints[i] = False
                        # Record the step when we finished visiting all waypoints
                        if final_waypoint_steps[i] is None:
                            final_waypoint_steps[i] = driver.envs[i]._env.unwrapped.step_count
                    latent, _ = carry
                    carry = (latent, agent._split(jax.device_put(acts, agent.policy_sharded)))
        if epsilon != 0.0:
            eps_acts = {}
            eps_mask = np.random.random(size=driver.length) < epsilon
            for k in acts.keys():
                act_space = driver.act_space[k]
                random_acts = np.stack([act_space.sample() for _ in range(driver.length)])
                eps_acts[k] = np.where(eps_mask, random_acts, acts[k])
            latent, _ = carry
            acts, outs, carry = eps_acts, outs, (latent, agent._split(jax.device_put(eps_acts, agent.policy_sharded)))
            
        outs['control_deter'] = control_outs['deter']
        outs['control_stoch'] = control_outs['stoch']
        # use actions from the policy as previous action to the control agent
        control_carry = (control_carry[0], carry[1])
        return acts, outs, carry

    driver.callbacks = []
    driver.on_step(log_step)
    driver(policy, episodes=num_episodes)

    return episodes_data



def main(argv=None):
    parsed, other = embodied.Flags(
        logdir="",
        collect_n_episodes=300,
        episodes_per_waypoint=10,
        num_waypoints=3,
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)

    # Set seeds
    set_seed(config.seed)
    
    # Create the environment once
    env = make_env(config, index=0)
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    env.close()

    checkpoint = embodied.Checkpoint()
    checkpoint.agent = agent
    checkpoint.step = embodied.Counter()
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        checkpoint.load(ckpt_path, keys=["agent", "step"])
        fns = [bind(make_env, config, index=i, estimate_belief=True) for i in range(config.run.num_envs)]
        # disable parallel env creation as we need easy 
        # access to the envs for waypoint episode rollouts
        driver = embodied.Driver(fns, False)
        
        # Generate waypoints once and collect episodes in batches
        waypoint_rng = np.random.RandomState(42)
        waypoint_episodes = []
        
        while len(waypoint_episodes) < parsed.collect_n_episodes:
            waypoints = driver.envs[0]._env.unwrapped.generate_random_waypoints(parsed.num_waypoints, rng=waypoint_rng)
            num_episodes = min(parsed.episodes_per_waypoint, parsed.collect_n_episodes - len(waypoint_episodes))
            episodes = collect_rollouts(
                ckpt_path,
                config,
                driver,
                num_episodes,
                collect_only_rewards=False,
                waypoints=waypoints
            )
            waypoint_episodes.extend(episodes)
        
        noisy_episodes = collect_rollouts(
            ckpt_path,
            config,
            driver,
            parsed.collect_n_episodes,
            epsilon=0.25,
            collect_only_rewards=False
        )
        
        regular_episodes = collect_rollouts(
            ckpt_path,
            config,
            driver,
            parsed.collect_n_episodes, 
            collect_only_rewards=False
        )
        
        driver.close()
        
        ckpt_number = int(checkpoint.step)
        np.savez(
            str(logdir / f"episodes_{ckpt_number}.npz"),
            episodes=regular_episodes,
            noisy_episodes=noisy_episodes,
            waypoint_episodes=waypoint_episodes
        )

# Example usage
if __name__ == "__main__":
    main()