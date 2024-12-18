import argparse
import os
from collections import defaultdict
from random import choices, random

import gymnasium
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamerv3 import embodied
from dreamerv3.embodied.core import logger as embodied_logger
from gymnasium.vector import AsyncVectorEnv
from tqdm import tqdm

import powm.envs
from powm.algorithms.train_dreamer import VideoOutput


class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.rnn = nn.GRU(
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
    """
    Trajectory storing the observed action, observations, rewards, and
    termination flags.

    Arguments:
    - action_size: int
        The number of discrete actions for the environment.
    - observation_size: int
        The observation size for the environment.
    """

    def __init__(self, action_size, observation_size):

        self.action_size = action_size
        self.observation_size = observation_size
        self.observed = []
        self.terminal = False

    def add(self, action, reward, observation, terminal=False):
        """
        Adds a new action and its outcome to the trajectory.

        Arguments:
        - action: int
            The action played in the environment.
        - reward: float
            The reward obtained after playing this action.
        - observation: tensor
            The new observation.
        - terminal: bool
            Whether a terminal state has been reached.
        """
        assert not self.terminal

        one_hot = torch.zeros(self.action_size)
        if action is not None:
            one_hot[action] = 1.
        action = one_hot
        observation = torch.tensor(observation, dtype=torch.float32)

        if reward is not None:
            reward = torch.tensor([reward], dtype=torch.float)
        else:
            reward = torch.tensor([0.], dtype=torch.float)

        self.observed.append(torch.cat((action, observation, reward)))

        if terminal:
            self.terminal = True

    @property
    def num_transitions(self):
        """
        Number of stored transitions.
        """
        return len(self.observed) - 1

    def get_cumulative_reward(self, gamma=1.0):
        """
        Returns the cumulative reward, discounted by gamma.

        Arguments:
        - gamma: float
            The discount factor.

        Returns:
        - cumulative_return: float
            The (discounted) cumulative return.
        """
        return sum(o[-1] * gamma ** t for t, o in enumerate(self.observed[1:]))

    def get_last_observed(self, number=None):
        """
        Returns the last observables (the last action and new observation).
        Note that the reward is not comprised in the observables.

        Arguments:
        - number: int
            Number of last oberservations.

        Returns:
        - observed: tensor
            The last observation(s).
        """
        if number is None:
            return self.observed[-1][:-1]
        else:
            truncated = [obs[:-1] for obs in self.observed[- number:]]
            if len(truncated) < number:
                padding = []
                for _ in range(number - len(truncated)):
                    padding.append(torch.zeros(self.observed[-1].size(0) - 1))
                truncated = padding + truncated
            return torch.stack(truncated)

    def get_transitions(self):
        """
        Returns the last observables (the last action and new observation).
        Note that the reward is not comprised in the observables.

        Returns:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            The list of all transitions in the trajectory.
        """
        sequence = torch.stack(self.observed, dim=0)

        transitions = []
        for t in range(sequence.size(0) - 1):
            seq_bef = sequence[:t + 1, :-1]
            seq_aft = sequence[:t + 2, :-1]
            a = sequence[t + 1, :self.action_size]
            o = sequence[t + 1, self.action_size:-1]
            r = sequence[t + 1, -1]
            if a.sum() == 0:
                a = None
                r = None
            else:
                a = a.argmax()
                r = r.item()
            d = self.terminal and t == sequence.size(0) - 2
            transitions.append((seq_bef, a, r, o, d, seq_aft))

        return transitions

class ReplayBuffer:
    """
    Replay Buffer storing transitions.

    Arguments:
    - capacity: int
        The number of transitions that can be stored in the replay buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.last = 0
        self.count = 0
    
    @property
    def ready(self):
        return self.count >= 1

    @property
    def is_full(self):
        """
        Whether the replay buffer is full.
        """
        return self.capacity == self.count

    def add_transition(self, transition):
        """
        Adds a transition to the replay buffer.

        Arguments:
        - transition: tuple (eta, a, r', o', d', eta')
            The transition to be added.
        """
        if self.count < self.capacity:
            self.buffer.append(transition)
            self.count += 1
        else:
            self.buffer[self.last] = transition
            self.last = (self.last + 1) % self.capacity

    def add(self, trajectory):
        """
        Adds all transitions of a trajectory to the replay buffer.

        Arguments:
        - trajectory: Trajectory
            The trajectory to be added.
        """
        assert isinstance(trajectory, Trajectory)
        for transition in trajectory.get_transitions():
            self.add_transition(transition)

    def sample(self, number):
        """
        Samples transitions from the replay buffer.

        Arguments:
        - number: int
            The number of transitions to be sampled.

        Returns:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            The list of transitions sampled in the replay buffer.
        """
        return choices(self.buffer, k=number)

class DRQNAgent(nn.Module):
    """Deep Recurrent Q-Network reinforcement learning agent."""
    def __init__(self, action_size, observation_size, hidden_size=256, num_layers=1, device='cpu'):
        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size
        self.device = device

        # Initialize Q network and target Q network
        self.Q = RNN(observation_size + action_size, action_size, hidden_size, num_layers).to(device)
        self.Q_tar = RNN(observation_size + action_size, action_size, hidden_size, num_layers).to(device)

        # Add default hidden shape
        self.hidden_shape = (num_layers, 1, hidden_size)

    def policy(self, observations, state=None, epsilon=0.0):
        """Returns actions and new states for a batch of observations.
        
        Args:
            observations: Current observations
            state: Tuple of (previous_action, hidden_state). If None, initializes new state.
            epsilon: Exploration parameter
        """
        batch_size = observations.shape[0] if isinstance(observations, torch.Tensor) else len(observations)
        
        # Convert observations to tensor if needed
        if not isinstance(observations, torch.Tensor):
            observations = torch.tensor(observations, dtype=torch.float32).to(self.device)
        
        # Initialize or unpack state
        if state is None:
            prev_actions = torch.zeros(batch_size, self.action_size).to(self.device)
            hidden = torch.zeros(self.hidden_shape[0], batch_size, self.hidden_shape[2]).to(self.device)
        else:
            prev_actions, hidden = state

        # Combine observation with previous action
        network_input = torch.cat([prev_actions, observations], dim=-1).unsqueeze(1)

        with torch.no_grad():
            q_values, new_hidden = self.Q(network_input, hidden)

        # Handle epsilon-greedy exploration
        if random() < epsilon:
            actions = np.random.randint(0, self.action_size, size=batch_size)
        else:
            actions = q_values.squeeze(1).argmax(dim=1).cpu().numpy()

        # Prepare next state (convert actions to one-hot for next state)
        next_actions = torch.zeros(batch_size, self.action_size).to(self.device)
        next_actions[range(batch_size), actions] = 1

        return actions, (next_actions, new_hidden)

    def init_state(self, batch_size=1):
        """Initialize agent state with batch size."""
        prev_actions = torch.zeros(batch_size, self.action_size).to(self.device)
        hidden = torch.zeros(self.hidden_shape[0], batch_size, self.hidden_shape[2]).to(self.device)
        return (prev_actions, hidden)
    
    def _targets(self, transitions, gamma):
        """
        Computes from a set of transitions, a list of inputs sequences,
        of masks indicating the considered time steps, and targets for those
        time steps computed with the target network.

        Arguments:
        - transitions: list of tuples (eta, a, r', o', d', eta')
            List of transitions
        - gamma: float:
            Discount factor for the targets

        Returns:
        - inputs: list
            List of histories
        - targets: list
            List of target values
        - masks: list
            List of time steps for which the loss should be computed
        """
        inputs, targets, masks = [], [], []

        # TODO: no loop but use padding
        for seq_bef, a, r, o, d, seq_aft in transitions:

            target = torch.tensor(r).to(self.device)

            if not d:
                with torch.no_grad():
                    Q_next, _ = self.Q_tar(seq_aft.unsqueeze(1).to(self.device))

                target += gamma * Q_next[-1, 0, :].max()

            target = target.view(1, -1)
            mask = torch.zeros(seq_bef.size(0), self.action_size,
                               dtype=torch.bool)
            mask[-1, a] = True

            seq_bef = seq_bef.to(self.device)
            inputs.append(seq_bef)
            targets.append(target)
            masks.append(mask)

        return inputs, targets, masks

    def _loss(self, inputs, targets, masks):
        """
        Computes the MSE loss between the predictions at the time steps
        specified by the masks and the targets.

        Arguments:
        - inputs: list
            List of histories
        - targets: list
            List of target values
        - masks: list
            List of time steps for which the loss should be computed

        Returns:
        - loss: float
            The loss resulting from the predictions and targets
        """
        inputs = nn.utils.rnn.pad_sequence(inputs)
        targets = nn.utils.rnn.pad_sequence(targets)
        masks = nn.utils.rnn.pad_sequence(masks)

        outputs, _ = self.Q(inputs)

        return F.mse_loss(outputs.transpose(0, 1)[masks.transpose(0, 1)],
                          targets.transpose(0, 1).flatten())
            
    def init_hidden(self, batch_size):
        return torch.zeros(self.hidden_shape[0], batch_size, self.hidden_shape[2]).to(self.device)

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

def build_vec_env(env_name, env_kwargs, seed, num_envs=4):
    def make_env(idx):
        def _init():
            env = gymnasium.make(env_name, render_mode="rgb_array", **env_kwargs)
            env = gymnasium.wrappers.TransformObservation(
                env, 
                lambda obs: obs['vector'], 
                observation_space=env.observation_space.spaces['vector']
            )
            env.reset(seed=seed + idx)
            return env
        return _init
    
    return AsyncVectorEnv([make_env(i) for i in range(num_envs)])

def train_drqn_agent(env_name, env_kwargs, agent, config, logger, ckpt_path):
    num_envs = config.train.num_envs
    env = build_vec_env(env_name, env_kwargs, seed=config.seed, num_envs=num_envs)
    buffer = ReplayBuffer(config.train.buffer_capacity)
    
    optim = torch.optim.Adam(
        list(agent.Q.parameters()), 
        lr=config.train.learning_rate
    )
    
    num_transitions = 0
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episodes = defaultdict(embodied.Agg)
    
    should_log = embodied.when.Every(config.train.log_every)
    should_train = embodied.when.Every(config.train.train_every)
    should_save = embodied.when.Every(config.train.save_every)
    should_update_target = embodied.when.Every(config.train.target_period)
    max_steps = config.train.steps

    # Initialize environment and trajectories
    current_obs = env.reset()[0]  # Shape: (num_envs, obs_dim)
    trajectories = [
        Trajectory(env.single_action_space.n, env.single_observation_space.shape[0]) 
        for _ in range(num_envs)
    ]
    for i in range(num_envs):
        trajectories[i].add(None, None, current_obs[i])
   
    agent_states = agent.init_state(num_envs)  # Now returns (prev_actions, hidden)
    dones = [False] * num_envs
    should_save(logger.step) # So we don't save 0th checkpoint
    
    for _ in tqdm(range(max_steps)):
        logger.step.increment()
        frames = env.render()  # Shape: (num_envs, H, W, C)
        episodes[0].add("policy_log_image", frames[0], agg="stack")  # Only log first env

        # Environment interaction
        with torch.no_grad():
            obs_batch = torch.tensor(current_obs, dtype=torch.float32).to(agent.device)
            actions, agent_states = agent.policy(obs_batch, agent_states, epsilon=config.train.epsilon)
            
        o_next, rewards, terminated, truncated, _ = env.step(actions)
        
        # Update trajectories
        for i in range(num_envs):
            if dones[i]:
                continue
            trajectories[i].add(actions[i], rewards[i], o_next[i], terminal=terminated[i])

        current_obs = o_next
        dones = [terminated[i] or truncated[i] for i in range(num_envs)]
        # Handle completed episodes
        for i in range(num_envs):
            if dones[i]:
                # Add trajectory to buffer and log stats
                score = trajectories[i].get_cumulative_reward(gamma=config.train.gamma)
                length = trajectories[i].num_transitions

                result = episodes[i].result()
                logger.add({
                    'score': score,
                    'length': length,
                }, prefix='episode')
                
                epstats.add(result)
                episodes[i].reset()

                buffer.add(trajectories[i])
                num_transitions += trajectories[i].num_transitions

                # Reset trajectory and state for this environment
                trajectories[i] = Trajectory(env.single_action_space.n, env.single_observation_space.shape[0])
                trajectories[i].add(None, None, current_obs[i])
                
                # Reset both previous action and hidden state for this environment
                agent_states[0][i].zero_()  # Reset previous action
                agent_states[1][:, i].zero_()  # Reset hidden state
        # Training step every N environment steps
        if buffer.ready and should_train(logger.step):
            for _ in range(config.train.num_gradient_steps):
                transitions = buffer.sample(config.train.batch_size)
                inputs, targets, masks = agent._targets(
                    transitions,
                    config.train.gamma,
                )
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
    vec_env = build_vec_env(config.env.name, 
                            yaml.YAML(typ="safe").load(config.env.kwargs) or {}, 
                            config.seed, 
                            config.train.num_envs)
    action_size = vec_env.action_space[0].n
    observation_size = vec_env.observation_space.shape[1]
    vec_env.close()

    # Create agent
    agent = DRQNAgent(
        action_size=action_size,
        observation_size=observation_size,
        hidden_size=config.drqn.hidden_size,
        num_layers=config.drqn.num_layers,
        device=config.device,
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
    main()