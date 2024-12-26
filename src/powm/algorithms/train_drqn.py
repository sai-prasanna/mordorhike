from random import choices, random

import gymnasium
import numpy as np
import ruamel.yaml as yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from dreamerv3 import embodied
from dreamerv3.embodied.core import logger as embodied_logger

import powm.envs
from powm.algorithms.train_dreamer import VideoOutput


class GRU(nn.Module):
    """
    Gated Recurrent Unit.

    Arguments:
    - input_size: int
        The input size.
    - hidden_size: int
        The hidden state size.
    - num_layers: int
        The number of layers.
    - output_size: int
        The output size.
    """
    num_states = 1

    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.rnn = nn.GRU(input_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x, h=None):
        """
        Computes the forward pass on the sequence using the hidden state if
        provided.

        Arguments:
        - x: tensor
            Input sequence
        - h: tuple of tensor
            Initial hidden state

        Returns:
        - x: tensor
            Output sequence
        - h: tuple of tensor
            Final hidden state
        """
        if h is not None:
            h, = h
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, (h,)


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

        if reward is not None:
            reward = torch.tensor([reward], dtype=torch.float)
        else:
            reward = torch.tensor([0.], dtype=torch.float)
        observation = torch.from_numpy(observation)

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


class DRQN(nn.Module):
    """
    Deep Recurrent Q-Network reinforcement learning agent.

    Arguments:
    - cell: str
        The name of the recurrent cell.
    - action_size: int
        The number of discrete actions in the environment.
    - observation_size: int
        The dimension of the observation in the environment.
    - **network_kwargs: dict
        Additional arguments for the recurrent cell.
    """

    def __init__(self, action_size, observation_size, num_layers, hidden_size,  device='cuda',):
        super().__init__()
        self.action_size = action_size
        self.observation_size = observation_size
        self.device = device

        # Initialize Q network and target Q network
        input_size = action_size + observation_size
        self.Q = GRU(
            input_size=input_size,
            output_size=action_size,
            num_layers=num_layers,
            hidden_size=hidden_size
        )
        self.Q_tar = GRU(
            input_size=input_size,
            output_size=action_size,
            num_layers=num_layers,
            hidden_size=hidden_size
        )

    def eval_rollout(self, environment, num_rollouts):
        """
        Evaluates the (discounted) cumulative return over a certain number of
        rollouts.

        Arguments:
        - environment: Environment
            The environment on which the agent is evaluated.
        - num_rollouts: int
            The number of episodes over which the returns are averaged.

        Returns:
        - mean_return: float
            The average empirical cumulative return
        - mean_disc_return: float
            The average empirical discounted cumulative return
        """
        sum_returns, disc_returns = 0.0, 0.0

        for _ in range(num_rollouts):

            trajectory, = self.play(environment, epsilon=0.0)

            sum_returns += trajectory.get_cumulative_reward()
            disc_returns += trajectory.get_cumulative_reward(gamma=0.99)

        mean_return = sum_returns / num_rollouts
        mean_disc_return = disc_returns / num_rollouts

        return mean_return, mean_disc_return

    def play(
        self,
        environment,
        epsilon,
        return_hiddens=False,
        return_beliefs=False,
    ):
        """
        Plays a trajectory in the environment with the current policy of
        the agent and some noise.

        Arguments:
        - environment: Environment
            The environment on which to play.
        - epsilon: float
            The exploration rate at each time step.
        - return_hiddens: bool
            Whether to return the hidden states along with the trajectory.
        - return_beliefs: bool
            Whether to return the beliefs along with the trajectory.

        Returns:
        - trajectory: Trajectory
            The trajectory resulting from the interaction with the environment.
        - hiddens: list
            The list of flattened hidden states at each time step of the trajectory.
        - beliefs: list
            The list of beliefs at each time step of the trajectory.
        """
        hiddens, beliefs = [], []

        o, _ = environment.reset()
        trajectory = Trajectory(
            environment.action_size,
            environment.observation_size,
        )

        trajectory.add(None, None, o)

        hidden_states = None
        terminated = False
        truncated = False
        while not (terminated or truncated):

            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(self.device)
            with torch.no_grad():
                values, hidden_states = self.Q(tau_t, hidden_states)

            if return_hiddens:
                hiddens.append(hidden_states[0].detach().cpu().flatten().clone())

            if return_beliefs:
                beliefs.append(environment.get_belief())

            if random() < epsilon:
                a = environment.action_space.sample()
            else:
                a = values.flatten().argmax().item()

            o, r, terminated, truncated, _ = environment.step(a)

            trajectory.add(a, r, o, terminal=terminated)


        return_values = (trajectory,)

        if return_hiddens:
            return_values += (hiddens,)
        if return_beliefs:
            return_values += (beliefs,)

        return return_values

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

            target = torch.tensor(r, device=self.device)

            if not d:
                with torch.no_grad():
                    Q_next, _ = self.Q_tar(seq_aft.unsqueeze(1).to(self.device))

                target += gamma * Q_next[-1, 0, :].max()

            target = target.view(1, -1)
            mask = torch.zeros(seq_bef.size(0), self.action_size,
                             dtype=torch.bool, device=self.device)
            mask[-1, a] = True

            inputs.append(seq_bef.to(self.device))
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

    def train_rollout(
        self,
        environment,
        num_episodes,
        batch_size,
        learning_rate,
        num_gradient_steps,
        target_period,
        eval_period,
        num_rollouts,
        epsilon,
        buffer_capacity,
        fill_buffer=False,
    ):
        """
        Trains the reinforcement learning agent on the specified environment

        Arguments:
        - environment: Environment
            The environment on which to train the agent.
        - logger: function
            The function to call for logging the training statistics.
        - num_episodes: int
            The number of episodes to generate
        - batch_size: int
            The number of transitions in each minibatch
        - learning_rate: float
            The learning rate used in the Adam optimizer
        - num_gradient_steps: int
            The number of gradients steps made at each episode
        - target_period: int
            The period at which the target is updated in number of episodes
        - eval_period: the period at which the network is evaluated in number
            of episodes
        - num_rollouts: int
            The number of episodes for the evaluation
        - epsilon: float
            The exploration rate
        """
        optim = torch.optim.Adam(self.Q.parameters(), lr=learning_rate)
        num_transitions = 0

        # Initialise replay buffer
        buffer = ReplayBuffer(buffer_capacity)

        # Eventually fill replay buffer
        if fill_buffer:
            while not buffer.is_full:
                trajectory, = self.play(environment, epsilon=1.0)
                buffer.add(trajectory)
                

        for episode in range(num_episodes):

            # Evaluate and save weights
            if episode % eval_period == 0:
                mean_return, mean_disc_return = self.eval(
                    environment,
                    num_rollouts,
                )

                print(f'Episode {episode:04d}')
                print(
                    {'return': mean_return, 'disc_return': mean_disc_return}
                )

                print(
                    {
                        'train/episode': episode,
                        'train/return': mean_return,
                        'train/disc_return': mean_disc_return,
                        'train/num_transitions': num_transitions,
                    }
                )


            # Update target
            if episode % target_period == 0:
                self.Q_tar.load_state_dict(self.Q.state_dict())

            # Generate trajectory
            trajectory, = self.play(environment, epsilon=epsilon)
            buffer.add(trajectory)
            num_transitions += trajectory.num_transitions

            # Optimize Q-network
            for _ in range(num_gradient_steps):

                transitions = buffer.sample(batch_size)

                inputs, targets, masks = self._targets(
                    transitions,
                    gamma=0.99,
                )
                loss = self._loss(inputs, targets, masks)

                optim.zero_grad()
                loss.backward()
                optim.step()

        # Log and save last results
        mean_return, mean_disc_return = self.eval(environment, num_rollouts)

        print('Final evaluation')
        print({'return': mean_return, 'disc_return': mean_disc_return})

        print({'train/episode': episode,
                'train/return': mean_return,
                'train/disc_return': mean_disc_return,
                'train/num_transitions': num_transitions})




def train_drqn_agent(env, agent: DRQN, config, logger, ckpt_path):
    
    buffer = ReplayBuffer(config.train.buffer_capacity)

    optim = torch.optim.Adam(
        list(agent.Q.parameters()), 
        lr=config.train.learning_rate
    )
    agg = embodied.Agg()
    epstats = embodied.Agg()
    episode = embodied.Agg()

    should_log = embodied.when.Every(config.train.log_every)
    should_train = embodied.when.Every(config.train.train_every)
    should_save = embodied.when.Every(config.train.save_every)
    should_update_target = embodied.when.Every(config.train.target_period)
    max_steps = config.train.steps
    
    should_save(logger.step) # So we don't save 0th checkpoint
    while logger.step.value < max_steps:
        
        o, _ = env.reset()
        trajectory = Trajectory(env.action_space.n, env.observation_space.shape[0])
        trajectory.add(None, None, o)
        hidden_states = None
        terminated = False
        truncated = False
        while not (terminated or truncated):
            frame  = env.render()
            episode.add("policy_log_image", frame, agg="stack")
            tau_t = trajectory.get_last_observed().view(1, 1, -1).to(agent.device)
            with torch.no_grad():
                values, hidden_states = agent.Q(tau_t, hidden_states)

            if random() < config.train.epsilon:
                a = env.action_space.sample()
            else:
                a = values.flatten().argmax().item()
            o, r, terminated, truncated, _ = env.step(a)
            logger.step.increment()
            trajectory.add(a, r, o, terminal=terminated)
            buffer_ready = buffer.count > 0
            if buffer_ready and should_train(logger.step):
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
            if buffer_ready and should_update_target(logger.step):
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
    env.close()

def make_logger(logdir: embodied.Path, config):
    loggers = [
        embodied_logger.TerminalOutput(),
        embodied_logger.JSONLOutput(logdir, "metrics.jsonl"),
        embodied_logger.TensorBoardOutput(logdir),
        VideoOutput(logdir, fps=8),
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
    env = gymnasium.make(env_name, render_mode="rgb_array", **env_kwargs)
    env = gymnasium.wrappers.TransformObservation(
        env, 
        lambda obs: obs['vector'], 
        observation_space=env.observation_space.spaces['vector']
    )
    env.reset(seed=seed)
    return env

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

        
    if (logdir / "config.yaml").exists():
        config = embodied.Config.load(logdir / "config.yaml")
        print("Loaded config from", logdir / "config.yaml")
    else:
        config.save(logdir / "config.yaml")
        print("Saved config to", logdir / "config.yaml")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    env_kwargs = yaml.YAML(typ="safe").load(config.env.kwargs) or {}
    env = build_env(config.env.name, env_kwargs, config.seed)
    
    action_size = env.action_space.n
    observation_size = env.observation_space.shape[0]

    agent = DRQN(
        action_size=action_size,
        observation_size=observation_size,
        device=device,
        num_layers=config.drqn.num_layers,
        hidden_size=config.drqn.hidden_size
    )
    agent.to(device)
    logger = make_logger(logdir, config)
    train_drqn_agent(env, agent, config, logger, logdir)

if __name__ == "__main__":
    main()