import random
from collections import defaultdict

import cv2
import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import tqdm

from powm.algorithms.train_drqn import DRQN, Trajectory, build_env
from powm.utils import set_seed


def collect_rollouts(
    agent: DRQN,
    config,
    num_episodes: int,
    epsilon: float = 0.0,
    collect_only_rewards: bool = False,
    waypoints: list = None,
    control_agent = None,
):
    """Collect rollouts from an agent.
    
    Args:
        config: Configuration object
        num_episodes: Number of episodes to collect
        epsilon: Exploration rate
        collect_only_rewards: If True, only collect rewards
        waypoints: If provided, list of waypoints to follow
        agent: Pre-created agent to use (if None, will create a new one)
        control_agent: Pre-created control agent to use (if None, will create a new one)
        
    Returns:
        List of episode data dictionaries
    """
    # Create environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    if not collect_only_rewards:
        env_kwargs["estimate_belief"] = True
    env = build_env(config.env.name, env_kwargs, config.seed)
    agent = agent.eval()
    
    episode_data = []
    rolled_out_episodes = 0
    
    while rolled_out_episodes < num_episodes:
        episode = defaultdict(list)
        
        o, infos = env.reset()
        
        trajectory = Trajectory(env.action_space.n, env.observation_space.shape[0])
        trajectory.add(None, None, o)

        hidden_states = None
        control_hidden_states = None if not collect_only_rewards else None
        terminated = False
        truncated = False
        
        # Reset waypoint tracking for new episode
        visited = None
        following_waypoints = waypoints is not None
        waypoints_steps = []
        
        while not (terminated or truncated):
            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(agent.device)
            with torch.no_grad():
                values, hidden_states = agent.Q(tau_t, hidden_states)
                if not collect_only_rewards:
                    _, control_hidden_states = control_agent.Q(tau_t, control_hidden_states)
            
            a = None
            # Get action based on current policy mode
            if following_waypoints:
                a, visited = env.unwrapped.get_waypoint_action(waypoints, visited)
                if visited is not None and len(waypoints_steps) < len(visited):
                    waypoints_steps.append(env.unwrapped.step_count)
                if a is None:  # Current waypoint reached or all waypoints visited
                    following_waypoints = False
            
            if a is None:
                # Use agent policy
                with torch.no_grad():
                    a = values.flatten().argmax().item()
                if epsilon != 0.0 and random.random() < epsilon:
                    a = env.action_space.sample()
            
            if not collect_only_rewards:
                # Extract latent states
                latent = hidden_states[0][:, 0].cpu().numpy().reshape(-1)
                control_latent = control_hidden_states[0][:, 0].cpu().numpy().reshape(-1)
                
                episode["state"].append(infos["state"])
                episode["belief"].append(infos["belief"])
                episode["latent"].append(latent)
                episode["control_latent"].append(control_latent)
                episode["obs"].append(o)
                episode["action"].append(a)
            
            o, r, terminated, truncated, infos = env.step(a)
            trajectory.add(a, r, o, terminal=terminated)
            episode["reward"].append(r)

            if terminated or truncated:
                if collect_only_rewards:
                    episode["reward"] = np.array(episode["reward"])
                else:
                    episode["state"] = np.array(episode["state"])
                    episode["obs"] = np.array(episode["obs"])
                    episode["action"] = np.array(episode["action"])
                    episode["reward"] = np.array(episode["reward"])
                    episode["latent"] = np.array(episode["latent"])
                    episode["control_latent"] = np.array(episode["control_latent"])
                    episode["belief"] = np.array(episode["belief"])
                    episode["success"] = np.array(terminated)
                    if waypoints is not None:
                        episode["waypoints_step"] = np.array(waypoints_steps)
                        episode["waypoints"] = waypoints
                episode_data.append(episode)
                rolled_out_episodes += 1
    
    env.close()
    return episode_data

def eval_checkpoint(checkpoint_path: str | None, config, num_episodes: int, epsilon: float = 0.0) -> list[float]:
    """Evaluate a checkpoint or random agent and return episode returns.
    
    Args:
        checkpoint_path: Path to checkpoint to evaluate, or None for random agent
        config: Configuration object
        num_episodes: Number of episodes to evaluate
        epsilon: Exploration rate for evaluation
        
    Returns:
        List of returns for each episode
    """
    # Create environment and agent
    env_kwargs = {}
    env = build_env(config.env.name, env_kwargs, config.seed)
    
    agent = DRQN(
        action_size=env.action_space.n,
        observation_size=env.observation_space.shape[0],
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(str(checkpoint_path))
        agent.load_state_dict(checkpoint["agent"])
    agent.eval()
    
    # Create control agent for rollouts
    control_agent = DRQN(
        action_size=env.action_space.n,
        observation_size=env.observation_space.shape[0],
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
    )
    control_agent.eval()
    
    # Collect evaluation episodes
    episodes = collect_rollouts(
        agent,
        config,
        num_episodes=num_episodes,
        epsilon=epsilon,
        control_agent=control_agent
    )
    
    # Calculate returns
    returns = [sum(ep["reward"]) for ep in episodes]
    return returns

def main(argv=None):
    from dreamerv3 import embodied
    parsed, other = embodied.Flags(
        logdir="",
        collect_n_episodes=300,
        episodes_per_waypoint=10,
        num_waypoints=3,
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)

    # Set seeds
    set_seed(config.seed)
    
    # Create environment for waypoint generation
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    env = build_env(config.env.name, env_kwargs, config.seed)
    
    # Create agents once
    agent = DRQN(
        action_size=env.action_space.n,
        observation_size=env.observation_space.shape[0],
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
        device=config.device,
    )
    control_agent = DRQN(
        action_size=env.action_space.n,
        observation_size=env.observation_space.shape[0],
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
        device=config.device,
    )
    
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])
        
        # Load checkpoint
        checkpoint = torch.load(str(ckpt_path), weights_only=False)
        agent.load_state_dict(checkpoint["agent"])

        # Collect rollouts
        with torch.no_grad():
            # Collect regular and noisy episodes
            regular_episodes = collect_rollouts(
                agent=agent,
                config=config,
                num_episodes=parsed.collect_n_episodes,
                control_agent=control_agent
            )
            noisy_episodes = collect_rollouts(
                agent=agent,
                config=config,
                num_episodes=parsed.collect_n_episodes,
                epsilon=0.25,
                control_agent=control_agent
            )
            # Collect waypoint episodes in batches
            waypoint_rng = np.random.RandomState(42)
            waypoint_episodes = []
            episodes_collected = 0
            
            while episodes_collected < parsed.collect_n_episodes:
                waypoints = env.unwrapped.generate_random_waypoints(parsed.num_waypoints, rng=waypoint_rng)
                num_episodes = min(parsed.episodes_per_waypoint, parsed.collect_n_episodes - episodes_collected)
                episodes = collect_rollouts(
                    agent=agent,
                    config=config,
                    num_episodes=num_episodes,
                    waypoints=waypoints,
                    control_agent=control_agent
                )
                waypoint_episodes.extend(episodes)
                episodes_collected += len(episodes)
            
            
        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{step}.npz",
            episodes=regular_episodes,
            noisy_episodes=noisy_episodes,
            waypoint_episodes=waypoint_episodes
        )
    
    env.close()


if __name__ == "__main__":
    main()
