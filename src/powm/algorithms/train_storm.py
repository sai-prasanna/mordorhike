import argparse
from collections import defaultdict, deque

import colorama
import gymnasium
import numpy as np
import ruamel.yaml as yaml
import torch
from dreamerv3 import embodied
from dreamerv3.embodied.core import logger as embodied_logger
from einops import rearrange
from storm_wm import agents, env_wrapper
from storm_wm.replay_buffer import ReplayBuffer
from storm_wm.sub_models.world_models import WorldModel
from storm_wm.utils import seed_np_torch
from tensorboardX import SummaryWriter
from tqdm import tqdm

from powm.algorithms.train_dreamer import VideoOutput
from powm.utils import set_seed


def make_logger(logdir: embodied.Path, config):
    loggers = [
        embodied_logger.TerminalOutput(),
        embodied_logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied_logger.TensorBoardOutput(logdir),
        VideoOutput(logdir, fps=20),
        embodied_logger.WandBOutput(
            wandb_init_kwargs=dict(
                project=config.wandb.project,
                group=logdir.parent.name,
                name=logdir.name,
                config=dict(config),
                dir=logdir,
            )
        ) if config.wandb.project else None,
    ]
    loggers = [x for x in loggers if x is not None]
    return embodied_logger.Logger(embodied.Counter(), loggers)

def build_single_env(env_name, env_kwargs, seed ,index=0):
    env_seed = hash((seed, index)) % (2 ** 32 - 1)
    if env_name.startswith("ALE/"):
        env = gymnasium.make(env_name, full_action_space=False, render_mode="rgb_array", frameskip=1)
        env.reset(seed=env_seed)
        env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=env_kwargs['skip'])
        env = gymnasium.wrappers.ResizeObservation(env, shape=env_kwargs['size'])
        env = env_wrapper.LifeLossInfo(env)
    else:
        env = gymnasium.make(env_name, render_mode="rgb_array", **env_kwargs)
        env.reset(seed=env_seed)
        env = gymnasium.wrappers.TransformObservation(env, lambda obs: obs['vector'], observation_space=env.observation_space.spaces['vector'])
    return env


def build_vec_env(env_name, num_envs, seed, env_kwargs):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, i):
        return lambda: build_single_env(env_name, env_kwargs, seed, i)
    env_fns = []
    env_fns = [lambda_generator(env_name, i) for i in range(num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)
    return vec_env


def train_world_model_step(replay_buffer: ReplayBuffer, world_model: WorldModel, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    metrics = world_model.update(obs, action, reward, termination)
    logger.add(metrics, prefix="train")


@torch.no_grad()
def world_model_imagine_data(replay_buffer: ReplayBuffer,
                             world_model: WorldModel, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat, logged_video = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
    )
    if logged_video is not None:
        logger.video("Imagine/predict_video", logged_video)
    return latent, action, None, None, reward_hat, termination_hat


def train_world_model_agent(env_name, env_kwargs, max_steps, num_envs,
                                  replay_buffer, world_model, agent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every, log_every, seed, logger, ckpt_path):

    vec_env = build_vec_env(env_name, num_envs=num_envs, seed=seed, env_kwargs=env_kwargs)
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    current_obs, _ = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)
    
    # Add frame collection for video logging

    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    
    should_log = embodied.when.Every(log_every)
    should_save = embodied.when.Every(save_every)
    should_train_agent = embodied.when.Every(train_agent_every_steps)
    should_train_world_model = embodied.when.Every(train_dynamics_every_steps)


    for total_steps in tqdm(range(max_steps//num_envs)):
        # Rest of your existing sampling code...
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()
                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()
                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action = agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )
            if world_model.encoder_type == "cnn":
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            else:
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B D -> B 1 D"))
            context_action.append(action)
        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info.get("life_loss", False)))
        done_flag = np.logical_or(done, truncated)
        frames = vec_env.render()
        
        for i in range(num_envs):
            episode = episodes[i]
            episode.add('score', reward[i], agg='sum')
            episode.add('length', 1, agg='sum')
            if i == 0:
                episode.add('policy_log_image', frames[i], agg='stack')
            if done_flag[i]:
                result = episode.result()
                # discounted return
                # compute the discounted return using gamma
                logger.add({
                    'score': result.pop('score'),
                    'length': result.pop('length'),
                }, prefix='episode')
                
                epstats.add(result)
                episode.reset()
            logger.step.increment()
        
        # update current_obs, current_info and sum_reward
        current_obs = obs
        
        if should_log(logger.step):
            logger.add(epstats.result(), prefix='epstats')
            logger.add(agg.result(), prefix='agg')
            logger.write()

        if replay_buffer.ready() and should_train_world_model(logger.step):
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger
            )

        if replay_buffer.ready() and should_train_agent(logger.step):
            if env_name.startswith("ALE/"):
                log_video = True
            else:
                log_video = False

            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger
            )

            metrics = agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
            )
            logger.add(metrics, prefix="train")
        
        if should_save(logger.step):
            print(colorama.Fore.GREEN + f"Saving model at total steps {logger.step}" + colorama.Style.RESET_ALL)
            torch.save({
                "world_model": world_model.state_dict(),
                "agent": agent.state_dict(),
                "step": logger.step.value,
            }, ckpt_path/f"checkpoint_{logger.step.value}.ckpt")



def build_world_model(conf, action_dim):
    return WorldModel(
        in_channels=conf.world_model.in_channels,
        action_dim=action_dim,
        transformer_max_length=conf.world_model.transformer_max_length,
        transformer_hidden_dim=conf.world_model.transformer_hidden_dim,
        transformer_num_layers=conf.world_model.transformer_num_layers,
        transformer_num_heads=conf.world_model.transformer_num_heads,
        transformer_positional_encoding=conf.world_model.transformer_positional_encoding,
        encoder_type=conf.world_model.encoder_type,
        input_dim=conf.world_model.input_dim,
        hidden_dim=conf.world_model.hidden_dim,
    ).cuda()


def build_agent(conf, action_dim):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.world_model.transformer_hidden_dim,
        num_layers=conf.agent.num_layers,
        hidden_dim=conf.agent.hidden_dim,
        action_dim=action_dim,
        gamma=conf.agent.gamma,
        lambd=conf.agent.lambd,
        entropy_coef=conf.agent.entropy_coef,
    ).cuda()


def main(argv=None):
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load configs
    configs = yaml.safe_load((embodied.Path(__file__).parent / "storm_config.yaml").read())
    parsed, other = embodied.Flags(configs=["defaults"]).parse_known(argv)
    config = embodied.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = embodied.Flags(config).parse(other)
    config = config.update(
        logdir=config.logdir.format(timestamp=embodied.timestamp()),
    )

    # Create and handle logdir
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    
    if (logdir / "config.yaml").exists():
        config = embodied.Config.load(logdir / "config.yaml")
        print("Loaded config from", logdir / "config.yaml")
    else:
        config.save(logdir / "config.yaml")
        print("Saved config to", logdir / "config.yaml")


    # set seed
    seed_np_torch(seed=config.seed)
    
    # Create logger
    logger = make_logger(logdir=logdir, config=config)

    # getting action_dim with dummy env
    dummy_env = build_single_env(config.env.name, env_kwargs=yaml.YAML(typ='safe').load(config.env.kwargs) or {}, seed=config.seed)
    action_dim = dummy_env.action_space.n

    # build world model and agent
    world_model = build_world_model(config, action_dim)
    agent = build_agent(config, action_dim)
    # print the total number of parameters
    world_model_params = sum(p.numel() for p in world_model.parameters())
    agent_params = sum(p.numel() for p in agent.parameters())
    total_params = world_model_params + agent_params
    print(f"Total number of world model parameters: {world_model_params}")
    print(f"Total number of agent parameters: {agent_params}")
    print(f"Total number of parameters: {total_params}")

    # build replay buffer
    if config.world_model.encoder_type == "cnn":
        obs_shape = (config.image_size, config.image_size, 3)
    else:
        obs_shape = (config.world_model.input_dim,)

    replay_buffer = ReplayBuffer(
        obs_shape=obs_shape,
        num_envs=config.train.num_envs,
        max_length=config.train.buffer_max_length,
        warmup_length=config.train.buffer_warm_up,
        store_on_gpu=config.replay_buffer_on_gpu
    )
    if config.train.demonstration_path:
        print(colorama.Fore.MAGENTA + f"loading demonstration trajectory from {config.train.demonstration_path}" + colorama.Style.RESET_ALL)
        replay_buffer.load_trajectory(path=config.train.demonstration_path)

    # Create checkpoint directory
    ckpt_path = logdir
    
    env_kwargs = yaml.YAML(typ='safe').load(config.env.kwargs)
    # train
    train_world_model_agent(
        env_name=config.env.name,
        env_kwargs=env_kwargs,
        num_envs=config.train.num_envs,
        max_steps=config.train.sample_max_steps,
        replay_buffer=replay_buffer,
        world_model=world_model,
        agent=agent,
        train_dynamics_every_steps=config.train.train_dynamics_every_steps,
        train_agent_every_steps=config.train.train_agent_every_steps,
        batch_size=config.train.batch_size,
        demonstration_batch_size=config.train.demonstration_batch_size if config.train.demonstration_path else 0,
        batch_length=config.train.batch_length,
        imagine_batch_size=config.train.imagine_batch_size,
        imagine_demonstration_batch_size=config.train.imagine_demonstration_batch_size if config.train.demonstration_path else 0,
        imagine_context_length=config.train.imagine_context_length,
        imagine_batch_length=config.train.imagine_batch_length,
        save_every=config.train.save_every,
        log_every=config.train.log_every,
        seed=config.seed,
        logger=logger,
        ckpt_path=ckpt_path
    )

if __name__ == "__main__":
    main()