import random
from collections import defaultdict

import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import tqdm

from powm.algorithms.train_drqn import DRQN, Trajectory, build_env
from powm.utils import set_seed


def collect_rollouts(agent, config, num_episodes, epsilon=0.0):
    episode_data = []

    # Create vectorized environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    env_kwargs["estimate_belief"] = True
    env = build_env(config.env.name, env_kwargs, config.seed)
    rolled_out_episodes = 0
    
    while rolled_out_episodes < num_episodes:
        episode = defaultdict(list)
        o, infos  = env.reset()
        trajectory = Trajectory(env.action_space.n, env.observation_space.shape[0])
        trajectory.add(None, None, o)


        hidden_states = None
        terminated = False
        truncated = False
        
        while not (terminated or truncated):
            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(agent.device)
            with torch.no_grad():
                values, hidden_states = agent.Q(tau_t, hidden_states)
                a = values.flatten().argmax().item()
            if epsilon != 0.0 and random.random() < epsilon:
                a = env.action_space.sample()
            latent = hidden_states[0][:, 0].cpu().numpy().reshape(1, 1, -1)
            
            
            episode["state"].append(infos["state"])
            episode["belief"].append(infos["belief"])
            episode["latent"].append(latent)
            episode["obs"].append(o)
            
            o, r, terminated, truncated, infos = env.step(a)
            trajectory.add(a, r, o, terminal=terminated)
            episode["action"].append(a)
            episode["reward"].append(r)

            if terminated or truncated:
                episode["state"] = np.array(episode["state"])
                episode["obs"] = np.array(episode["obs"])
                episode["action"] = np.array(episode["action"])
                episode["reward"] = np.array(episode["reward"])
                episode["latent"] = np.array(episode["latent"])
                episode["belief"] = np.array(episode["belief"])
                episode["success"] = np.array(terminated)
                
                episode_data.append(episode)
                rolled_out_episodes += 1
    env.close()
    return episode_data

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

    # Create agent
    env = build_env(config.env.name, yaml.YAML(typ="safe").load(config.env.kwargs) or {}, config.seed)
    action_size = env.action_space.n
    observation_size = env.observation_space.shape[0]

    agent = DRQN(
        action_size=action_size,
        observation_size=observation_size,
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
    )
    agent.to(config.device)
    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])
        checkpoint = torch.load(str(ckpt_path))
        agent.load_state_dict(checkpoint["agent"])
        agent.eval()

        # Collect rollouts
        with torch.no_grad():
            episodes = collect_rollouts(
                agent,
                config,
                num_episodes=parsed.collect_n_episodes
            )
            noisy_episodes = collect_rollouts(
                agent,
                config,
                num_episodes=parsed.collect_n_episodes,
                epsilon=0.25
            )

        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{step}.npz",
            episodes=episodes,
            noisy_episodes=noisy_episodes
        )


if __name__ == "__main__":
    main()
