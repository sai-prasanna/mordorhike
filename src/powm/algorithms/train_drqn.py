import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from random import random
import gymnasium
import argparse
import numpy as np
import torch
from collections import defaultdict
from tqdm import tqdm


from dreamerv3.embodied.core import logger as embodied_logger
from dreamerv3 import embodied
import ruamel.yaml as yaml

from powm.algorithms.train_dreamer import VideoOutput
import powm.envs

class RNNS:
    """RNN cell types"""
    gru = lambda input_size, output_size, hidden_size, num_layers: \
        RNN(input_size, output_size, hidden_size, num_layers, cell='gru')
    lstm = lambda input_size, output_size, hidden_size, num_layers: \
        RNN(input_size, output_size, hidden_size, num_layers, cell='lstm')

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, cell='gru'):
        super().__init__()
        self.rnn = getattr(nn, cell.upper())(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        x, h = self.rnn(x, h)
        return self.head(x), h

class Trajectory:
    def __init__(self, action_size, observation_size):
        self.action_size = action_size
        self.observation_size = observation_size
        self.reset()

    def reset(self):
        self.actions = []
        self.rewards = []
        self.observations = []
        self.terminals = []

    def add(self, action, reward, observation, terminal=False):
        if action is not None:
            self.actions.append(action)
        if reward is not None:
            self.rewards.append(reward)
        if observation is not None:
            self.observations.append(observation)
        if terminal:
            self.terminals.append(True)
        else:
            self.terminals.append(False)

    @property
    def num_transitions(self):
        return len(self.actions)

    def get_cumulative_reward(self, gamma=1.0):
        return sum(r * (gamma ** t) for t, r in enumerate(self.rewards))

    def get_sequence_before(self, t):
        """Returns the sequence of observations and actions before time t"""
        seq = []
        for tau in range(t + 1):
            o_tau = torch.tensor(self.observations[tau], dtype=torch.float32)
            if tau < t:
                a_tau = torch.zeros(self.action_size)
                a_tau[self.actions[tau]] = 1
                seq.append(torch.cat([o_tau, a_tau]))
            else:
                seq.append(torch.cat([o_tau, torch.zeros(self.action_size)]))
        return torch.stack(seq)

    def get_sequence_after(self, t):
        """Returns the sequence of observations and actions after time t"""
        seq = []
        for tau in range(t + 1, len(self.observations)):
            o_tau = torch.tensor(self.observations[tau], dtype=torch.float32)
            if tau < len(self.actions):
                a_tau = torch.zeros(self.action_size)
                a_tau[self.actions[tau]] = 1
                seq.append(torch.cat([o_tau, a_tau]))
            else:
                seq.append(torch.cat([o_tau, torch.zeros(self.action_size)]))
        return torch.stack(seq)

    def get_last_observed(self):
        """Returns the last observation"""
        return torch.tensor(
            self.observations[-1],
            dtype=torch.float32,
        ).view(1, -1)

    def is_terminal(self):
        return any(self.terminals)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.trajectories = []
        self.position = 0

    def add(self, trajectory):
        if len(self.trajectories) < self.capacity:
            self.trajectories.append(trajectory)
        else:
            self.trajectories[self.position] = trajectory
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Returns a batch of transitions sampled from all trajectories
        in the buffer.
        """
        transitions = []
        for _ in range(batch_size):
            # Sample trajectory
            trajectory = np.random.choice(self.trajectories)
            # Sample time step
            t = np.random.randint(trajectory.num_transitions)
            # Get sequences before and after t
            seq_bef = trajectory.get_sequence_before(t)
            seq_aft = trajectory.get_sequence_after(t)
            # Get action, reward and terminal flag
            a = trajectory.actions[t]
            r = trajectory.rewards[t]
            d = trajectory.terminals[t]
            # Add transition to batch
            transitions.append((seq_bef, a, r, trajectory.observations[t + 1], d, seq_aft))

        return transitions

    @property
    def ready(self):
        return len(self.trajectories) > 0

    @property
    def is_full(self):
        return len(self.trajectories) == self.capacity

class DRQNAgent(nn.Module):
    """Deep Recurrent Q-Network reinforcement learning agent."""
    def __init__(self, action_size, observation_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size

        # Initialize Q network and target Q network
        self.Q = RNN(observation_size + action_size, action_size, hidden_size, num_layers)
        self.Q_tar = RNN(observation_size + action_size, action_size, hidden_size, num_layers)

    def forward(self, observation, hidden=None, epsilon=0.0):
        """Returns action and new hidden state based on current policy."""
        tau_t = torch.cat([
            torch.tensor(observation, dtype=torch.float32),
            torch.zeros(self.action_size)
        ]).view(1, 1, -1)

        with torch.no_grad():
            q_values, hidden = self.Q(tau_t, hidden)

        if random() < epsilon:
            action = np.random.randint(self.action_size)
        else:
            action = q_values.flatten().argmax().item()

        return action, hidden

    def _targets(self, transitions, gamma):
        """Computes target values for Q-learning."""
        inputs, targets, masks = [], [], []

        for seq_bef, a, r, o, d, seq_aft in transitions:
            target = torch.tensor(r, dtype=torch.float32)

            if not d:
                with torch.no_grad():
                    Q_next, _ = self.Q_tar(seq_aft.unsqueeze(1))
                target += gamma * Q_next[-1, 0, :].max()

            target = target.view(1, -1)
            mask = torch.zeros(seq_bef.size(0), self.action_size, dtype=torch.bool)
            mask[-1, a] = True

            inputs.append(seq_bef)
            targets.append(target)
            masks.append(mask)

        return inputs, targets, masks

    def _loss(self, inputs, targets, masks):
        """Computes the MSE loss between predictions and targets."""
        inputs = nn.utils.rnn.pad_sequence(inputs)
        targets = nn.utils.rnn.pad_sequence(targets)
        masks = nn.utils.rnn.pad_sequence(masks)

        outputs, _ = self.Q(inputs)
        return F.mse_loss(
            outputs.transpose(0, 1)[masks.transpose(0, 1)],
            targets.transpose(0, 1).flatten()
        )

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

def build_env(env_name, env_kwargs, seed):
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = gymnasium.wrappers.TransformObservation(env, lambda obs: obs['vector'], observation_space=env.observation_space.spaces['vector'])
    env.reset(seed=seed)
    return env

def train_drqn_agent(env_name, env_kwargs, agent, config, logger, ckpt_path):
    env = build_env(env_name, env_kwargs, seed=config.seed)
    buffer = ReplayBuffer(config.train.buffer_capacity)
    
    optim = torch.optim.Adam(
        list(agent.Q.parameters()), 
        lr=config.train.learning_rate
    )
    
    num_transitions = 0
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episode = embodied.Agg()
    
    should_log = embodied.when.Every(config.train.log_every)
    should_train = embodied.when.Every(config.train.train_every)
    should_save = embodied.when.Every(config.train.save_every)
    should_update_target = embodied.when.Every(config.train.target_period)
    max_steps = config.train.steps

    # Initialize environment and trajectory
    
    current_obs = env.reset()[0]
    trajectory = Trajectory(env.action_space.n, env.observation_space.shape[0])
    trajectory.add(None, None, current_obs)
   
    hidden = None
    done = False
    
    for _ in tqdm(range(max_steps)):
        logger.step.increment()
        frames = env.render()
        episode.add("policy_log_image", frames, agg="stack")
        # Environment interaction
        with torch.no_grad():
            a, hidden = agent.forward(current_obs, hidden, epsilon=config.train.epsilon)
        o_next, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        
        trajectory.add(a, r, o_next, terminal=terminated)
        current_obs = o_next

        # If episode ended, reset environment and start new trajectory
        if done:
            # Log episode stats
            score = trajectory.get_cumulative_reward(gamma=config.train.gamma)
            length = trajectory.num_transitions

            result = episode.result()
            logger.add({
                'score': score,
                'length': length,
            }, prefix='episode')
            
            epstats.add(result)
            episode.reset()

            buffer.add(trajectory)
            num_transitions += trajectory.num_transitions

            # Reset environment and trajectory
            current_obs = env.reset()[0]
            trajectory = Trajectory(env.action_space.n, env.observation_space.shape[0])
            trajectory.add(None, None, current_obs)
            hidden = None

        # Training step every N environment steps
        if buffer.ready and should_train(logger.step):
            for _ in range(config.train.num_gradient_steps):
                transitions = buffer.sample(config.train.batch_size)
                inputs, targets, masks = agent._targets(transitions, config.train.gamma)
                loss = agent._loss(inputs, targets, masks)

                optim.zero_grad()
                loss.backward()
                optim.step()

                logger.add({'train/loss': loss.item()})
        
        if should_update_target(logger.step):
            agent.Q_tar.load_state_dict(agent.Q.state_dict())


        if should_save(logger.step):
            checkpoint = {
                'agent': agent.state_dict(),
                'optimizer': optim.state_dict(),
                'step': logger.step.value,
                'config': config
            }
            checkpoint_path = ckpt_path / f'checkpoint_{logger.step.value}.ckpt'
            torch.save(checkpoint, checkpoint_path)
            torch.save(checkpoint, "checkpoint.ckpt")

        if should_log(logger.step):
            logger.add(epstats.result(), prefix='epstats')
            logger.add(agg.result(), prefix='agg')
            logger.write()


    env.close()

def main(argv=None):
    # Load configs
    configs = yaml.safe_load((embodied.Path(__file__).parent / "drqn_configs.yaml").read())
    parsed, other = embodied.Flags(configs=["defaults"]).parse_known(argv)
    config = embodied.Config(configs["defaults"])
    for name in parsed.configs:
        config = config.update(configs[name])
    config = embodied.Flags(config).parse(other)
    config = config.update(
        logdir=config.logdir.format(timestamp=embodied.timestamp()),
    )

    # Create logdir
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    
    if (logdir / "config.yaml").exists():
        config = embodied.Config.load(logdir / "config.yaml")
        print("Loaded config from", logdir / "config.yaml")
    else:
        config.save(logdir / "config.yaml")
        print("Saved config to", logdir / "config.yaml")

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Create logger
    logger = make_logger(logdir=logdir, config=config)

    # Create environment to get dimensions
    env = build_env(config.env.name, config.seed, yaml.safe_load(config.env.kwargs))
    action_size = env.action_space.n
    observation_size = env.observation_space.shape[0]
    env.close()

    # Create agent
    agent = DRQNAgent(
        action_size=action_size,
        observation_size=observation_size,
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
    )

    # Train agent
    train_drqn_agent(
        env_name=config.env.name,
        env_kwargs=yaml.YAML(typ='safe').load(config.env.kwargs),
        agent=agent,
        config=config,
        logger=logger,
        ckpt_path=logdir
    )

if __name__ == "__main__":
    main(argv=["--logdir", "experiments/drqn"])