import re
from collections import defaultdict

import numpy as np
import recall2imagine
from recall2imagine import embodied

from powm.algorithms.train_r2i import make_envs
from powm.utils import set_seed


def collect_rollouts(
    checkpoint_path: str | None,
    config,
    env,
    num_episodes: int,
    epsilon: float = 0.0,
    collect_only_rewards: bool = False,
    waypoints: list = None,
):
    """Collect rollouts from a R2I agent.
    
    Args:
        checkpoint_path: Path to checkpoint to load, or None for random agent
        config: Configuration object
        env: Environment to run in
        num_episodes: Number of episodes to collect
        epsilon: Exploration rate
        collect_only_rewards: If True, only collect rewards
        waypoints: If provided, list of waypoints to follow
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

    # Initialize waypoint tracking
    visited_waypoints = [None] * len(driver._env)
    following_waypoints = [True] * len(driver._env) if waypoints is not None else None
    final_waypoint_steps = [None] * len(driver._env)

    def log_step(tran, worker):
        nonlocal waypoints, visited_waypoints, following_waypoints, final_waypoint_steps
        current_episode = worker_episodes[worker]
        # Always collect rewards for score calculation
        current_episode["reward"].append(tran["reward"])
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
        if not collect_only_rewards:
            current_episode["state"].append(info["state"])
            current_episode["belief"].append(info["belief"])
            # Store raw deter and stoch temporarily
            latent_keys = ["deter", "stoch", "hidden", "logit", "control_deter", "control_stoch", "control_hidden"]
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
            
            if collect_only_rewards:
                episodes_data.append({
                    'reward': np.array(current_episode["reward"]),
                })
            else:
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
                    current_episode[key] = v[0]
                    
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
                episode_data = {
                    **{k: np.array(v) for k, v in current_episode.items()},
                    'latent': latents,
                    'control_latent': control_latents,
                    "success": tran["is_terminal"]
                }
                # Add waypoint data if using waypoints
                if waypoints is not None:
                    episode_data["final_waypoint_step"] = np.array(final_waypoint_steps[worker])
                    episode_data["waypoints"] = waypoints

                episodes_data.append(episode_data)

            # Reset waypoint tracking for this worker
            visited_waypoints[worker] = None
            if following_waypoints is not None:
                following_waypoints[worker] = True
            final_waypoint_steps[worker] = None
            current_episode.clear()

    control_state = None
    def policy(obs, state, **kwargs):
        nonlocal control_state, visited_waypoints, following_waypoints, final_waypoint_steps
        if state and 'control_deter' in state[0][0].keys():
            del state[0][0]['control_deter']
            del state[0][0]['control_stoch']
            del state[0][0]['control_hidden']
        acts, state = agent.policy(obs, state, **kwargs)
        _, control_state = control_agent.policy(obs, control_state, **kwargs)
        
        # Handle waypoint navigation
        if waypoints is not None:
            for i in range(len(driver._env)):
                gym_env = env._envs[i].env.env.env.env._env.unwrapped
                if following_waypoints[i]:
                    waypoint_action, visited_waypoints[i] = gym_env.get_waypoint_action(
                        waypoints, visited_waypoints[i]
                    )
                    if waypoint_action is not None:
                        one_hot_waypoint_action = np.zeros_like(acts['action'][i])
                        one_hot_waypoint_action[waypoint_action] = 1
                        acts['action'] = acts['action'].copy()
                        acts['action'][i] = one_hot_waypoint_action
                    else:
                        following_waypoints[i] = False
                        # Record the step when we finished visiting all waypoints
                        if final_waypoint_steps[i] is None:
                            final_waypoint_steps[i] = gym_env.step_count
            
            (prev_latent, _), task_state, expl_state = state
            state = (prev_latent, acts['action']), task_state, expl_state

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
        episodes_per_waypoint=3,
        num_waypoints=3,
    ).parse_known(argv)
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    # Disable parallel processes because we need to access the environment
    # for waypoint generation and policy rollouts
    config =config.update({"envs.parallel": "none"})

    # Set seeds
    set_seed(config.seed)

    # Create environment and agent
    env = make_envs(config, estimate_belief=True)
    step = embodied.Counter()
    for ckpt_path in sorted(ckpt_paths, key=lambda x: int(x.stem.split("_")[-1])):
        # Collect regular and noisy episodes
        regular_episodes = collect_rollouts(
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
        # Generate waypoints and collect episodes in batches
        waypoint_rng = np.random.RandomState(4)
        waypoint_episodes = []
        
        while len(waypoint_episodes) < parsed.collect_n_episodes:
            waypoints = env._envs[0].env.env.env.env._env.unwrapped.generate_random_waypoints(
                parsed.num_waypoints, 
                rng=waypoint_rng
            )
            num_episodes = min(
                parsed.episodes_per_waypoint, 
                parsed.collect_n_episodes - len(waypoint_episodes)
            )
            episodes = collect_rollouts(
                ckpt_path,
                config,
                env,
                num_episodes,
                collect_only_rewards=False,
                waypoints=waypoints
            )
            waypoint_episodes.extend(episodes)
        # Save episode data
        ckpt_number = int(step)
        np.savez(
            f"{parsed.logdir}/episodes_{ckpt_number}.npz",
            episodes=regular_episodes,
            noisy_episodes=noisy_episodes,
            waypoint_episodes=waypoint_episodes
        )

    env.close()


if __name__ == "__main__":
    main()