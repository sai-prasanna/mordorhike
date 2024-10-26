import numpy as np
import pomdp_py
import pomdp_py.algorithms

from powm.envs.mordor import MordorHike


# Define the state for MordorHike
class MordorHikeState(pomdp_py.State):
    def __init__(self, state):
        self.state = state

    def __hash__(self):
        return hash(tuple(self.state))

    def __eq__(self, other):
        return isinstance(other, MordorHikeState) and np.all(self.state == other.state)

    def __str__(self):
        return f"MordorHikeState(state={self.state})"

# Define actions for MordorHike
class MordorHikeAction(pomdp_py.Action):
    def __init__(self, idx):
        self.idx = idx

    def __hash__(self):
        return hash(self.idx)

    def __eq__(self, other):
        return isinstance(other, MordorHikeAction) and self.idx == other.idx

    def __str__(self):
        return f"MordorHikeAction(idx={self.idx})"

# Define observations for MordorHike
class MordorHikeObservation(pomdp_py.Observation):
    def __init__(self, observation):
        self.observation = np.round(observation, 1)

    def __hash__(self):
        return hash(tuple(self.observation))

    def __eq__(self, other):
        return isinstance(other, MordorHikeObservation) and np.all(self.observation == other.observation)

    def __str__(self):
        return f"MordorHikeObservation(position={self.observation})"

import scipy
# Observation model for MordorHike
from scipy.stats import norm


class ObservationModel(pomdp_py.ObservationModel):
    def __init__(self, env: MordorHike):
        self.env = env
        # Removed noise parameter

    def probability(self, observation, next_state, action):
        # Calculate the probability of observing the given observation from the next state
        observation = observation.observation
        env_state = next_state.state
        obs_gaussian_mu = self.env.obs_gaussain_mu(np.array([env_state]))[0]
        cov = np.diag(np.full(obs_gaussian_mu.shape[-1], self.env.obs_std**2))
        return scipy.stats.multivariate_normal(mean=obs_gaussian_mu, cov=cov).pdf(observation)

    def sample(self, next_state, action):
        # Sample an observation based on the next state and action using env properties
        env_state = next_state.state
        obs = self.env.observation(np.array([env_state]))[0]
        return MordorHikeObservation(obs)

# Transition model for MordorHike
class TransitionModel(pomdp_py.TransitionModel):
    def __init__(self, env: MordorHike):
        self.env = env

    def probability(self, next_state, state, action):
        # Use the dynamics of the MordorHike environment to determine the transition probability
        env_state = state.state
        pred_next_state = self.env.dynamics_mu(np.array([env_state[:2]]), np.array([env_state[2:]]), np.array([action.idx]))[0]
        prob = 0.0
        if np.all(next_state.state == pred_next_state):
            prob += self.env.action_failure_prob
        cov = np.diag(np.full(pred_next_state[:2].shape[-1], self.env.translate_std**2))
        translate_prob = scipy.stats.multivariate_normal(mean=pred_next_state[:2], cov=cov).pdf(next_state.state[:2])
        if self.env.rotate_kappa is not None:
            rotate_prob = scipy.stats.vonmises(pred_next_state[2], self.env.rotate_kappa, next_state.state[2]).pdf(next_state.state[2])
        prob += (1 - self.env.action_failure_prob) * translate_prob * rotate_prob
        return prob
    
    def sample(self, state, action):
        # Sample the next state based on the current state and action
        env_state = state.state
        new_state = self.env.dynamics(np.array([env_state]), np.array([action.idx]))[0]
        return MordorHikeState(new_state)

# Reward model for MordorHike
class RewardModel(pomdp_py.RewardModel):
    def __init__(self, env: MordorHike):
        self.env = env

    def sample(self, state, action, next_state):
        next_env_state = next_state.state
        terminated = self.env._terminal(np.array([next_env_state]))[0]
        reward = 0 if terminated else self.env._altitude(np.array([next_env_state]))[0]
        return reward


class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = {MordorHikeAction(idx) for idx in range(4)}
    def __init__(self, env: MordorHike):
        self.env = env

    def rollout(self, state, *args):
        """Treating this PolicyModel as a rollout policy"""
        return MordorHikeAction(np.random.randint(4))

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS

# Define the MordorHike POMDP problem
class MordorHikeProblem(pomdp_py.POMDP):
    def __init__(self, env: MordorHike, init_state: MordorHikeState, init_belief: pomdp_py.Particles):
        self.gym_env = env
        agent = pomdp_py.Agent(
            init_belief,
            PolicyModel(env),
            TransitionModel(env),
            ObservationModel(env),
            RewardModel(env),
        )
        env = pomdp_py.Environment(
            init_state,
            TransitionModel(env),
            RewardModel(env),
        )

        super().__init__(agent, env, name="MordorHikeProblem")

    @staticmethod
    def create(env: MordorHike):
        init_state = MordorHikeState(env.state)
        init_belief = pomdp_py.Particles([MordorHikeState(p) for p in env.particles])
        return MordorHikeProblem(env, init_state, init_belief)



def run_planner(problem: MordorHikeProblem, planner):
    done = False
    while not done:
        action = planner.plan(problem.agent)


        print(f"True state: {problem.env.state}")
        print(f"Action: {action}")

        # reward = problem.env.state_transition(action, execute=True)
        obs, reward, terminated, truncated, _ = problem.gym_env.step(action.idx)
        done = terminated or truncated
        # use planner belief instead of true belief
        problem.gym_env.particles = np.array([p.state for p in problem.agent.cur_belief])

        problem.gym_env.render()
        real_observation = MordorHikeObservation(obs)
        problem.agent.update_history(action, real_observation)
        # Update the belief. If the planner is POMCP, planner.update
        # also automatically updates agent belief.
        planner.update(problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)

# Example usage
def main():
    env = MordorHike.hard(render_mode="human", lateral_action="strafe", estimate_belief=True, num_particles=5000)
    env.reset(seed=12)
    problem = MordorHikeProblem.create(env)
    pomcp = pomdp_py.POMCP( 
        max_depth=10,
        discount_factor=0.99,
        num_sims=4000,
        exploration_const=50,
        rollout_policy=problem.agent.policy_model,
        show_progress=True,
        pbar_update_interval=500,
    )
    run_planner(problem, pomcp)

    # Implement planner and testing logic here...

if __name__ == "__main__":
    main()