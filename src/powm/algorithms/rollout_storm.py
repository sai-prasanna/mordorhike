import random
from collections import defaultdict
import re
import numpy as np
import torch
from storm_wm.utils import seed_np_torch
from storm_wm.sub_models.world_models import WorldModel
from storm_wm import agents, env_wrapper
import gymnasium
import ruamel.yaml as yaml
from powm.algorithms.train_storm import build_single_env, build_vec_env

def collect_rollouts(logger, agent, world_model, config, num_episodes):
    episodes_data = []
    worker_episodes = defaultdict(lambda: defaultdict(list))
    scores = []
    lengths = []
    returns = []
    
    # Create vectorized environment
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs)
    vec_env = build_vec_env(config.env.name, config.train.num_envs, config.seed, env_kwargs)
    
    current_obs, _ = vec_env.reset()
    context_obs = defaultdict(lambda: [])
    context_action = defaultdict(lambda: [])

    def log_step(obs, action, reward, done, worker):
        current_episode = worker_episodes[worker]
        
        # Collect episode data
        current_episode["observations"].append(obs)
        current_episode["action"].append(action)
        current_episode["rewards"].append(reward)
        
        # Store latent states
        if len(context_obs[worker]) > 0:
            context_latent = world_model.encode_obs(torch.cat(context_obs[worker], dim=1))
            model_context_action = np.stack(context_action[worker], axis=1)
            model_context_action = torch.Tensor(model_context_action).cuda()
            prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                context_latent, model_context_action)
            current_episode["latents"].append(
                torch.cat([prior_flattened_sample, last_dist_feat], dim=-1).cpu().numpy())
        
        if done:
            # Calculate episode statistics
            score = sum(current_episode["rewards"])
            length = len(current_episode["rewards"])
            scores.append(score)
            lengths.append(length)
            
            # Calculate discounted return
            rewards = np.array(current_episode["rewards"])
            discount_factor = config.agent.gamma
            discounts = discount_factor ** np.arange(len(rewards))
            discounted_return = np.sum(rewards * discounts)
            returns.append(discounted_return)
            
            # Get latent trajectories
            latents = np.array(current_episode["latents"])
            
            # Compute predictions using world model
            predictions = world_model.imagine_data(
                agent,
                torch.Tensor(current_episode["observations"]).cuda(),
                torch.Tensor(current_episode["action"]).cuda(),
                imagine_batch_size=1,
                imagine_batch_length=5,
                log_video=False
            )
            
            # Store episode data
            episodes_data.append({
                'observations': np.array(current_episode["observations"]),
                'actions': np.array(current_episode["action"]),
                'rewards': rewards,
                'latents': latents,
                'predictions': predictions
            })
            
            # Clear episode data
            current_episode.clear()
            context_obs[worker] = []
            context_action[worker] = []

    # Collect episodes
    for _ in range(num_episodes):
        done = False
        while not done:
            if len(context_action[0]) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(context_obs[0], dim=1))
                model_context_action = np.stack(context_action[0], axis=1)
                model_context_action = torch.Tensor(model_context_action).cuda()
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(
                    context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

            context_obs[0].append(torch.Tensor(current_obs).cuda().unsqueeze(1) / 255)
            context_action[0].append(action)
            
            obs, reward, done, truncated, _ = vec_env.step(action)
            log_step(current_obs, action, reward, done or truncated, 0)
            current_obs = obs
            done = done or truncated

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
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs)
    dummy_env = build_single_env(config.env.name, env_kwargs, config.seed, 0)
    action_dim = dummy_env.action_space.n
    dummy_env.close()
    
    # Build world model and agent
    from storm_wm.train import build_world_model, build_agent
    world_model = build_world_model(config, action_dim)
    agent = build_agent(config, action_dim)
    
    # Load checkpoints
    ckpt_paths = sorted([f for f in logdir.glob("world_model_*.pth")])
    for ckpt_path in ckpt_paths:
        step = int(ckpt_path.stem.split("_")[-1])
        world_model.load_state_dict(torch.load(str(ckpt_path)))
        agent.load_state_dict(torch.load(str(logdir / f"agent_{step}.pth")))
        
        # Create logger
        logger = embodied.Logger(parsed.logdir, step, config)
        
        # Collect rollouts
        episodes = collect_rollouts(
            logger,
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
        
        logger.close()

if __name__ == "__main__":
    main(["--logdir", "experiments/storm_easy_256_wo_sinu3/1"])
