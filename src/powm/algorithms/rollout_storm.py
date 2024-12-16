from collections import defaultdict
import numpy as np
from collections import deque

from einops import rearrange
import torch
from storm_wm.utils import seed_np_torch

import ruamel.yaml as yaml
from powm.algorithms.train_storm import build_single_env, build_vec_env

def collect_rollouts(agent, world_model, config, num_episodes):
    episodes_data = []
    current_episode = defaultdict(list)
    scores = []
    lengths = []
    returns = []
    
    # Create vectorized environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    env_kwargs["estimate_belief"] = True
    vec_env = build_vec_env(config.env.name, 1, config.seed, env_kwargs)
    
    current_obs, info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    
    IMAGINE_STEPS = 5
    
    # Collect episodes
    for _ in range(num_episodes):
        done = False
        while not done:
            # Regular episode collection (unchanged)
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
                current_latent = None
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                    context_latent, model_context_action)
                current_latent = torch.cat([prior_flattened_sample, last_dist_feat], dim=-1)
                action = agent.sample_as_env_action(
                    current_latent,
                    greedy=False
                )

            if world_model.encoder_type == "cnn":
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            else:
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B D -> B 1 D"))
            context_action.append(action)
            current_episode["state"].append(info["state"])
            current_episode["belief"].append(info["belief"])
            obs, reward, terminated, truncated, info = vec_env.step(action)
            done = terminated or truncated
            
            current_episode["obs"].append(current_obs)
            current_episode["action"].append(action)
            current_episode["reward"].append(reward)
            if current_latent is not None:
                current_episode["latent"].append(current_latent.cpu().numpy())
            else:
                wm_dim = config.world_model.transformer_hidden_dim + 32*32
                current_episode["latent"].append(np.zeros((1, 1, wm_dim)))
            
            if done:
                rewards = np.array(current_episode["reward"])
                # Episode statistics (unchanged)
                score = sum(rewards)
                length = len(rewards)
                scores.append(score)
                lengths.append(length)
                
                discount_factor = config.agent.gamma
                discounts = discount_factor ** np.arange(len(rewards))
                discounted_return = np.sum(rewards * discounts)
                returns.append(discounted_return)
                
                # Prepare episode data for imagination
                if world_model.encoder_type == "cnn":
                    episode_obs_tensor = torch.Tensor(np.array(current_episode["obs"])).cuda()
                    episode_obs_tensor = rearrange(episode_obs_tensor, "T B H W C -> B T C H W")/255
                else:
                    episode_obs_tensor = torch.Tensor(np.array(current_episode["obs"])).cuda()
                    episode_obs_tensor = rearrange(episode_obs_tensor, "T B D -> B T D")
                
                episode_actions_tensor = torch.Tensor(np.array(current_episode["action"])).cuda()
                episode_actions_tensor = rearrange(episode_actions_tensor, "T B -> B T")
                
                # Initialize world model for imagination
                world_model.storm_transformer.reset_kv_cache_list(1, dtype=world_model.tensor_dtype)
                context_latent = world_model.encode_obs(episode_obs_tensor)
                imagined_trajectories = []
                last_latent = None
                
                # Incrementally build KV cache and imagine
                for t in range(len(current_episode["obs"]) - IMAGINE_STEPS):
                    _, _, _, last_latent, _ = world_model.predict_next(
                        context_latent[:, t:t+1],
                        episode_actions_tensor[:, t:t+1],
                        log_video=False
                    )
                    
                    # Store KV cache reference
                    kv_cache_list = world_model.storm_transformer.kv_cache_list.copy()
                    
                    # Imagine next IMAGINE_STEPS
                    imagined_obs = []
                    imagined_latent = last_latent
                    
                    for step in range(IMAGINE_STEPS):
                        true_action = episode_actions_tensor[:, t+step+1:t+step+2]
                        obs_hat, _, _, imagined_latent, _ = world_model.predict_next(
                            imagined_latent,
                            true_action,
                            log_video=True
                        )
                        imagined_obs.append(obs_hat)
                    
                    imagined_trajectories.append(torch.cat(imagined_obs, dim=1))
                    
                    # Reset KV cache to reference point
                    world_model.storm_transformer.kv_cache_list = kv_cache_list
                imagined_trajectories = torch.stack(imagined_trajectories, dim=1).cpu().float().numpy()
                
                # consistent shapes for recorded data
                current_episode["obs"] = np.array(current_episode["obs"])[:, 0]
                current_episode["action"] = np.array(current_episode["action"])[:, 0]
                current_episode["reward"] = np.array(current_episode["reward"])[:, 0]
                # timesteps x particles x latent_dim (some latents don't have particles like STORM)
                current_episode["latent"] = np.array(current_episode["latent"])[:, 0]
                # timesteps x particles x belief_dim
                current_episode["belief"] = np.array(current_episode["belief"])[:, 0]
                # remove batch dim, add particle dim as it is just 1 for STORM
                # Timesteps x particles x future_prediction_timesteps x obs_dim
                current_episode["obs_hat"] = np.array(imagined_trajectories)[0][:, np.newaxis, ...]
                current_episode["state"] = np.array(current_episode["state"])[:, 0]
                current_episode["success"] = terminated
                episodes_data.append(dict(current_episode))
                current_episode = defaultdict(list)
            
            current_obs = obs
    vec_env.close()
    return episodes_data

def main(argv=None):
    # Parse arguments similar to train_storm.py
    from dreamerv3 import embodied
    parsed, other = embodied.Flags(
        logdir="",
        metric_dir="eval",
    ).parse_known(argv)
    
    assert parsed.logdir, "Logdir is required"
    logdir = embodied.Path(parsed.logdir)
    config = embodied.Config.load(str(logdir / "config.yaml"))
    config = embodied.Flags(config).parse(other)
    
    # Set seeds
    seed_np_torch(seed=config.seed)
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs) or {}
    env_kwargs["estimate_belief"] = True
    dummy_env = build_single_env(config.env.name, env_kwargs, config.seed, 0)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    
    # Build world model and agent
    from powm.algorithms.train_storm import build_world_model, build_agent
    world_model = build_world_model(config, action_dim).eval()
    agent = build_agent(config, action_dim).eval()
    
    # Load checkpoints
    ckpt_paths = sorted([f for f in logdir.glob("world_model_*.pth")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])
        world_model.load_state_dict(torch.load(str(ckpt_path)))
        agent.load_state_dict(torch.load(str(logdir / f"agent_{step}.pth")))
        # Collect rollouts
        with torch.no_grad():
            episodes = collect_rollouts(
                agent,
                world_model, 
                config,
                110  # num episodes
            )
            
        # Save episode data
        np.savez(
            f"{parsed.logdir}/episodes_{step}.npz",
            episodes=episodes
        )
        

if __name__ == "__main__":
    main()