from collections import defaultdict

import numpy as np
import ruamel.yaml as yaml
import torch
from tqdm import tqdm

from powm.algorithms.train_drqn import DRQNAgent, build_vec_env, make_logger


def collect_rollouts(agent, config, num_episodes):
    episode_data = []

    # Create vectorized environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    env_kwargs["estimate_belief"] = True
    vec_env = build_vec_env(config.env.name, env_kwargs, config.seed, num_envs=config.train.num_envs)

    current_obs, infos  = vec_env.reset()
    agent_state = agent.init_state(batch_size=len(current_obs))
    
    rolled_out_episodes = 0
    current_dones = [False] * config.train.num_envs
    episodes = defaultdict(lambda: defaultdict(list))
    while rolled_out_episodes < num_episodes:
        with torch.no_grad():
            obs_tensor = torch.tensor(current_obs, dtype=torch.float32).to(agent.device)
            action, agent_state = agent.policy(obs_tensor, agent_state, epsilon=0.0)
        
        for i in range(len(current_obs)):
            if current_dones[i]:
                # This is required as the environment is reset after current_dones[i] is set to True
                # so the info would correspond to the last step of the episode instead of the reset step
                continue
            episodes[i]["state"].append(infos["state"][i])
            episodes[i]["belief"].append(infos["belief"][i])


        next_obs, reward, terminated, truncated, infos = vec_env.step(action)
        
        for i in range(len(current_obs)):
            if current_dones[i]:
                # If the episode is done, then the step would be reseting the environment, 
                # so action and reward would be invalid.
                continue
            episodes[i]["obs"].append(current_obs[i])
            episodes[i]["action"].append(action[i])
            episodes[i]["reward"].append(reward[i])
            
            latent = agent_state[1][:, i].cpu().numpy()
            # add particle dim and collapse the 
            # gru layer dim into single final dim
            latent = latent.reshape(latent.shape[0], 1, -1)
            episodes[i]["latent"].append(latent)
            
            done = terminated[i] or truncated[i]
            if done and rolled_out_episodes < num_episodes:
                episodes[i]["state"] = np.array(episodes[i]["state"])
                episodes[i]["obs"] = np.array(episodes[i]["obs"])
                episodes[i]["action"] = np.array(episodes[i]["action"])
                episodes[i]["reward"] = np.array(episodes[i]["reward"])
                episodes[i]["latent"] = np.array(episodes[i]["latent"])
                episodes[i]["belief"] = np.array(episodes[i]["belief"])
                episodes[i]["success"] = np.array(terminated[i])
                

                episode_data.append(episodes[i])
                prev_actions, prev_hiddens = agent_state
                
                worker_state = agent.init_state(batch_size=1)
                prev_actions[i] = worker_state[0][0]
                # for hidden state it's layers x batch size x hidden size
                # so we need to index the batch in the second dimension
                prev_hiddens[:, i] = worker_state[1][:, 0]
                agent_state = (prev_actions, prev_hiddens)

                episodes[i] = defaultdict(list)
                rolled_out_episodes += 1
        
        current_obs = next_obs
        current_dones = [d or t for d, t in zip(terminated, truncated)]
    vec_env.close()
    return episode_data

def main(argv=None):
    from dreamerv3 import embodied
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
        collect_n_episodes=110,
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create logger
    logger = make_logger(logdir=logdir, config=config)

    # Create agent
    vec_env = build_vec_env(config.env.name, yaml.YAML(typ="safe").load(config.env.kwargs) or {}, config.seed)
    action_size = vec_env.action_space[0].n
    observation_size = vec_env.observation_space.shape[1]
    vec_env.close()

    agent = DRQNAgent(
        action_size=action_size,
        observation_size=observation_size,
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
        device=config.device,
    )

    ckpt_paths = sorted([f for f in logdir.glob("checkpoint_*.ckpt")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])
        checkpoint = torch.load(str(ckpt_path))
        logger.step = embodied.Counter(checkpoint["step"])
        agent.load_state_dict(checkpoint["agent"])
        agent.eval()

        # Collect rollouts
        with torch.no_grad():
            episodes = collect_rollouts(
                agent,
                config,
                num_episodes=parsed.collect_n_episodes  # Number of episodes to collect
            )

        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{step}.npz",
            episodes=episodes
        )


if __name__ == "__main__":
    main(["--logdir", "/home/sai/Desktop/powm/experiments/train_drqn_1"])
