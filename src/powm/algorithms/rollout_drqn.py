import random
from collections import defaultdict

import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import tqdm

from powm.algorithms.train_drqn import DRQN, Trajectory, build_env
from powm.utils import set_seed


def collect_rollouts(
    checkpoint_path: str | None,
    config,
    num_episodes: int,
    epsilon: float = 0.0,
    collect_only_rewards: bool = False,
):
    """Collect rollouts from an agent.
    
    Args:
        checkpoint_path: Path to checkpoint to load, or None for random agent
        config: Configuration object
        num_episodes: Number of episodes to collect
        epsilon: Exploration rate
        collect_only_rewards: If True, only collect rewards
        
    Returns:
        List of episode data dictionaries. If collect_only_rewards=True, only reward data is collected.
    """
    # Create environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    if not collect_only_rewards:
        env_kwargs["estimate_belief"] = True
    env = build_env(config.env.name, env_kwargs, config.seed)
    
    # Setup agent
    agent = DRQN(
        action_size=env.action_space.n,
        observation_size=env.observation_space.shape[0],
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
        device=config.device,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(str(checkpoint_path), weights_only=False)
        agent.load_state_dict(checkpoint["agent"])
    agent = agent.eval()
    
    # Setup control agent if needed
    control_agent = None
    if not collect_only_rewards:
        control_agent = DRQN(
            action_size=env.action_space.n,
            observation_size=env.observation_space.shape[0],
            hidden_size=config.drqn.hidden_size,
            num_layers=config.drqn.num_layers,
            device=config.device,
        )
        control_agent = control_agent.eval()
    
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
        
        while not (terminated or truncated):
            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(agent.device)
            with torch.no_grad():
                values, hidden_states = agent.Q(tau_t, hidden_states)
                if not collect_only_rewards:
                    # Get control latents from control agent
                    _, control_hidden_states = control_agent.Q(tau_t, control_hidden_states)
                a = values.flatten().argmax().item()
            if epsilon != 0.0 and random.random() < epsilon:
                a = env.action_space.sample()
            
            if not collect_only_rewards:
                # Extract latent states
                latent = hidden_states[0][:, 0].cpu().numpy().reshape(1, 1, -1)
                control_latent = control_hidden_states[0][:, 0].cpu().numpy().reshape(1, 1, -1)
                
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
        control_agent,
        config,
        num_episodes=num_episodes,
        epsilon=epsilon
    )
    
    # Calculate returns
    returns = [sum(ep["reward"]) for ep in episodes]
    return returns

def main(argv=None):
    from dreamerv3 import embodied
    parsed, other = embodied.Flags(
        logdir="",
        collect_n_episodes=300,
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)

    # Set seeds
    set_seed(config.seed)
    
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])

        # Collect rollouts
        with torch.no_grad():
            episodes = collect_rollouts(
                checkpoint_path=ckpt_path,
                config=config,
                num_episodes=parsed.collect_n_episodes,
            )
            noisy_episodes = collect_rollouts(
                checkpoint_path=ckpt_path,
                config=config,
                num_episodes=parsed.collect_n_episodes,
                epsilon=0.25,
            )

        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{step}.npz",
            episodes=episodes,
            noisy_episodes=noisy_episodes
        )


if __name__ == "__main__":
    main()
