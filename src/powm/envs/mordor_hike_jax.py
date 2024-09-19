import time
from functools import partial

import cv2
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces


class MordorHikeJAX(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    @classmethod
    def easy(cls, **kwargs):
        return cls(occlude_dims=(0, 1), start_distribution="rotation", **kwargs)

    @classmethod
    def medium(cls, **kwargs):
        return cls(occlude_dims=(0, 1), start_distribution="rotation", **kwargs)

    @classmethod
    def hard(cls, **kwargs):
        return cls(occlude_dims=(0, 1), start_distribution="uniform", **kwargs)

    def __init__(
        self,
        occlude_dims=(0, 1),
        translate_step=0.1,
        rotate_step=np.pi / 2,
        translate_std=0.05,
        rotate_kappa=None,
        action_failure_prob=0.0,
        start_distribution="fixed",
        obs_std=0.1,
        render_mode=None,
        estimate_belief=False,
        num_particles=1000,
        effective_particle_threshold=0.5,
        render_size=(200, 200),
        lateral_action="strafe",
    ):
        self.render_mode = render_mode
        self.render_size = render_size
        self.background = None

        # Move parameters into self
        self.occlude_dims = jnp.array(occlude_dims)
        self.translate_step = translate_step
        self.rotate_step = rotate_step
        self.translate_std = translate_std
        self.rotate_kappa = rotate_kappa
        self.action_failure_prob = action_failure_prob
        self.start_distribution = start_distribution
        self.obs_std = obs_std
        self.estimate_belief = estimate_belief
        self.num_particles = num_particles
        self.effective_particle_threshold = effective_particle_threshold
        self.lateral_action = lateral_action

        # Add other parameters
        self.lower_bound = jnp.array((-1.0, -1.0))
        self.upper_bound = jnp.array((1.0, 1.0))
        self.fixed_start_pos = jnp.array((-0.8, -0.8))
        self.uniform_start_lower_bound = jnp.array((-1.0, -1.0))
        self.uniform_start_upper_bound = jnp.array((1.0, 0.5))
        self.goal_position = jnp.array((0.8, 0.8))
        self.mvn_1_mean = jnp.array((0.0, 0.0))
        self.mvn_1_cov = jnp.array(((0.005, 0.0), (0.0, 1.0)))
        self.mvn_2_mean = jnp.array((0.0, -0.8))
        self.mvn_2_cov = jnp.array(((1.0, 0.0), (0.0, 0.01)))
        self.mvn_3_mean = jnp.array((0.0, 0.8))
        self.mvn_3_cov = jnp.array(((1.0, 0.0), (0.0, 0.01)))
        self.slope = jnp.array((0.2, 0.2))

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.state = None
        self.path = []

    @partial(jax.jit, static_argnums=(0,))
    def _init_state(self, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)

        if self.start_distribution == "rotation":
            position = jnp.array(self.fixed_start_pos)
            num_rotations = int(2 * jnp.pi / self.rotate_step)
            theta = jax.random.randint(subkey1, (), 0, num_rotations) * self.rotate_step
        elif self.start_distribution == "uniform":
            position = jax.random.uniform(
                subkey1,
                (2,),
                minval=jnp.array(self.uniform_start_lower_bound),
                maxval=jnp.array(self.uniform_start_upper_bound),
            )
            num_rotations = int(2 * jnp.pi / self.rotate_step)
            theta = jax.random.randint(subkey2, (), 0, num_rotations) * self.rotate_step
        else:  # fixed
            position = jnp.array(self.fixed_start_pos)
            theta = 0.0

        return jnp.concatenate([position, jnp.array([theta])])

    @partial(jax.jit, static_argnums=(0,))
    def _observation(self, state, key):
        position = state[:2]
        alt = self._altitude(position)
        obs = jnp.concatenate([position, jnp.array([alt])], axis=-1)

        key, subkey = jax.random.split(key)
        obs += jax.random.normal(subkey, obs.shape) * self.obs_std
        obs = jnp.delete(obs, self.occlude_dims, axis=-1)
        return jnp.clip(obs, -1.0, 1.0)

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, action, key):
        key, subkey1, subkey2 = jax.random.split(key, 3)
        new_state = self._dynamics(state, action, subkey1)
        obs = self._observation(new_state, subkey2)
        reward = self._altitude(new_state[:2])
        terminated = (
            jnp.linalg.norm(new_state[:2] - jnp.array(self.goal_position))
            <= self.translate_step * 2
        )
        return new_state, obs, reward, terminated

    @partial(jax.jit, static_argnums=(0,))
    def _dynamics(self, state, action, key):
        position, theta = state[:2], state[2:]
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)

        def success(position, theta):
            def move_forward():
                forward_vector = (
                    jnp.concatenate([jnp.cos(theta), jnp.sin(theta)])
                    * self.translate_step
                )
                new_position = position + forward_vector * (1 - 2 * action)
                return new_position, theta

            def move_side():
                side_vector = (
                    jnp.concatenate([-jnp.sin(theta), jnp.cos(theta)])
                    * self.translate_step
                )
                new_position = position + side_vector * (1 - 2 * (action % 2))
                return new_position, theta

            def rotate():
                new_theta = jnp.mod(
                    theta + self.rotate_step * (1 - 2 * (action % 2)), 2 * jnp.pi
                )
                return position, new_theta

            if self.lateral_action == "strafe":
                position, theta = jax.lax.cond(action < 2, move_forward, move_side)
            elif self.lateral_action == "rotate":
                position, theta = jax.lax.cond(action < 2, move_forward, rotate)
            return position, theta

        def fail(position, theta):
            return position, theta

        position, theta = jax.lax.cond(
            jax.random.uniform(subkey1, ()) > self.action_failure_prob,
            success,
            fail,
            position,
            theta,
        )

        position += jax.random.normal(subkey2, position.shape) * self.translate_std

        if self.rotate_kappa is not None:
            theta += jax.random.vonmises(subkey3, 0, self.rotate_kappa, theta.shape)
            theta = jnp.mod(theta + 2 * jnp.pi, 2 * jnp.pi)
        position = jnp.clip(position, self.lower_bound, self.upper_bound)

        return jnp.concatenate([position, jnp.array(theta)], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _altitude(self, position):
        mountains = jnp.stack(
            [
                jax.scipy.stats.multivariate_normal.pdf(
                    position, self.mvn_1_mean, self.mvn_1_cov
                ),
                jax.scipy.stats.multivariate_normal.pdf(
                    position, self.mvn_2_mean, self.mvn_2_cov
                ),
                jax.scipy.stats.multivariate_normal.pdf(
                    position, self.mvn_3_mean, self.mvn_3_cov
                ),
            ]
        )

        altitude = jnp.max(mountains, axis=0)
        return -jnp.exp(-altitude) + (position @ jnp.array(self.slope)) - 0.02

    @partial(jax.jit, static_argnums=(0,))
    def _init_belief(self, key):
        N = self.num_particles
        key, subkey1, subkey2 = jax.random.split(key, 3)

        if self.start_distribution in ["fixed", "rotation"]:
            positions = jnp.tile(self.fixed_start_pos, (N, 1))
        else:  # uniform
            positions = jax.random.uniform(
                subkey1,
                (N, 2),
                minval=self.uniform_start_lower_bound,
                maxval=self.uniform_start_upper_bound,
            )

        num_rotations = int(2 * jnp.pi / self.rotate_step)
        if self.start_distribution == "fixed":
            thetas = jnp.zeros(N)
        else:
            thetas = (
                jax.random.randint(subkey2, (N,), 0, num_rotations) * self.rotate_step
            )

        particles = jnp.column_stack((positions, thetas))
        weights = jnp.ones(N) / N

        return particles, weights

    @partial(jax.jit, static_argnums=(0,))
    def _update_belief(self, particles, weights, action, obs, key):
        if action is not None:
            key, subkey = jax.random.split(key)
            subkeys = jax.random.split(subkey, particles.shape[0])
            particles = jax.vmap(lambda p, k: self._dynamics(p, action, k))(
                particles, subkeys
            )

        key, subkey = jax.random.split(key)
        subkeys = jax.random.split(subkey, particles.shape[0])
        particle_observations = jax.vmap(lambda p, k: self._observation(p, k))(
            particles, subkeys
        )

        likelihood = jnp.prod(
            jax.scipy.stats.norm.pdf(
                obs,
                loc=particle_observations,
                scale=self.obs_std,
            ),
            axis=-1,
        )

        weights = weights * likelihood
        weights = weights / jnp.sum(weights)  # Normalize weights

        n_eff = 1 / jnp.sum(jnp.square(weights))

        def resample(args):
            particles, weights, key = args
            key, subkey = jax.random.split(key)
            indices = jax.random.choice(
                subkey,
                self.num_particles,
                shape=(self.num_particles,),
                p=weights,
                replace=True,
            )
            return (
                particles[indices],
                jnp.ones(self.num_particles) / self.num_particles,
                key,
            )

        def keep_current(args):
            return args

        particles, weights, key = jax.lax.cond(
            n_eff < self.effective_particle_threshold * self.num_particles,
            resample,
            keep_current,
            (particles, weights, key),
        )

        return particles, weights

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.key = jax.random.PRNGKey(seed)
        self.key, subkey = jax.random.split(self.key)
        self.state = self._init_state(subkey)
        self.key, subkey = jax.random.split(self.key)
        obs = self._observation(self.state, subkey)

        info = {}
        if self.estimate_belief:
            self.key, subkey = jax.random.split(self.key)
            self.particles, self.weights = self._init_belief(subkey)
            self.key, subkey = jax.random.split(self.key)
            self.particles, self.weights = self._update_belief(
                self.particles, self.weights, None, obs, subkey
            )
            info["belief"] = (np.array(self.particles), np.array(self.weights))

        # Start tracking path
        self.path = [self.state[:2].tolist()]

        return np.asarray(obs), info

    def step(self, action):
        self.key, subkey = jax.random.split(self.key)
        self.state, obs, reward, done = self._step(self.state, action, subkey)

        info = {}
        if self.estimate_belief:
            self.key, subkey = jax.random.split(self.key)
            self.particles, self.weights = self._update_belief(
                self.particles, self.weights, action, obs, subkey
            )
            info["belief"] = (self.particles, self.weights)

        # Add current position to path
        self.path.append(self.state[:2])

        return np.asarray(obs), float(reward), bool(done), False, info

    def render(self):
        if self.render_mode is None:
            return

        # Create or get cached background
        if self.background is None:
            self.background = self._create_background()

        # Move computations to numpy for rendering
        img = self.background.copy()

        # Draw path
        if len(self.path) > 1:
            path_pixels = self._world_to_pixel(self.path)
            cv2.polylines(img, [np.asarray(path_pixels)], False, (0, 0, 255), 2)

        # Draw start and goal
        start = tuple(map(int, self._world_to_pixel(self.fixed_start_pos)))
        goal = tuple(map(int, self._world_to_pixel(self.goal_position)))
        cv2.circle(img, start, 5, (0, 255, 0), -1)
        cv2.circle(img, goal, 5, (255, 0, 0), -1)

        if self.estimate_belief:
            for particle, weight in zip(list(self.particles), list(self.weights)):
                pos = tuple(map(int, self._world_to_pixel(particle[:2])))
                size = int(5 + 45 * weight)
                cv2.circle(img, pos, size, (0, 165, 255), 1)

                direction = (
                    int(10 * np.cos(particle[2])),
                    int(-10 * np.sin(particle[2])),
                )
                cv2.line(
                    img,
                    pos,
                    (pos[0] + direction[0], pos[1] + direction[1]),
                    (0, 165, 255),
                    1,
                )

        # Draw actual position and direction
        state = np.array(self.state)
        pos = tuple(map(int, self._world_to_pixel(state[:2])))
        cv2.circle(img, pos, 7, (0, 0, 255), -1)
        direction = (int(15 * np.cos(state[2])), int(-15 * np.sin(state[2])))
        cv2.line(
            img, pos, (pos[0] + direction[0], pos[1] + direction[1]), (0, 0, 255), 2
        )

        if self.render_mode == "rgb_array":
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.render_mode == "human":
            cv2.imshow("Mordor Hike", img)
            cv2.waitKey(1)

    def _create_background(self):
        height, width = self.render_size
        x = jnp.linspace(self.lower_bound[0], self.upper_bound[0], width)
        y = jnp.linspace(self.lower_bound[1], self.upper_bound[1], height)
        X, Y = jnp.meshgrid(x, y)
        positions = jnp.stack([X, Y], axis=-1)
        Z = self._altitude(positions.reshape(-1, 2)).reshape(height, width)

        # Normalize Z to 0-255 range
        Z_norm = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(jnp.uint8)

        # Create color map
        color_map = cv2.applyColorMap(np.array(Z_norm), cv2.COLORMAP_VIRIDIS)

        # Draw contours
        levels = jnp.linspace(Z.min(), Z.max(), 10)
        for level in levels:
            contours = cv2.findContours(
                np.array((Z >= level).astype(jnp.uint8)),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
            cv2.drawContours(color_map, contours, -1, (255, 255, 255), 1)

        return color_map

    @partial(jax.jit, static_argnums=(0,))
    def _world_to_pixel(self, coord):
        coord = jnp.asarray(coord)
        x = jnp.round(
            (coord[..., 0] - self.lower_bound[0])
            / (self.upper_bound[0] - self.lower_bound[0])
            * self.render_size[1]
        ).astype(jnp.int32)
        # Flip y-axis for image coordinates
        y = jnp.round(
            (
                1
                - (coord[..., 1] - self.lower_bound[1])
                / (self.upper_bound[1] - self.lower_bound[1])
            )
            * self.render_size[0]
        ).astype(jnp.int32)
        return jnp.stack([x, y], axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def _rollout(self, state, actions, key):
        def step(carry, action):
            state, key, reward_sum = carry
            key, subkey = jax.random.split(key)
            new_state, obs, reward, terminated = self._step(state, action, subkey)
            return (new_state, key, reward_sum + reward), (
                new_state,
                reward,
                terminated,
            )

        (final_state, _, total_reward), (states, rewards, terminated) = jax.lax.scan(
            step, (state, key, 0.0), actions
        )
        return total_reward, states, rewards, terminated

    @partial(jax.jit, static_argnums=(0, 3, 4, 5, 6, 7))
    def cem_plan(
        self,
        state,
        key,
        n_iterations=5,
        n_samples=1000,
        n_elite=100,
        horizon=50,
        discount=0.99,
    ):
        # Initialize probabilities for each action (4 actions)
        probs = jnp.ones((horizon, 4)) / 4

        def iteration(carry, _):
            probs, key = carry
            key, subkey1, subkey2 = jax.random.split(key, 3)

            # Sample actions from multinomial distribution
            actions = jax.random.categorical(subkey1, probs, shape=(n_samples, horizon))

            subkeys = jax.random.split(subkey2, n_samples)
            _, _, rewards, terminated = jax.vmap(
                lambda a, k: self._rollout(state, a, k)
            )(actions, subkeys)

            # Calculate discounted rewards
            discount_factors = discount ** jnp.arange(horizon)
            masked_rewards = jnp.where(terminated, 0, rewards)
            discounted_rewards = jnp.sum(masked_rewards * discount_factors, axis=1)

            elite_idx = jnp.argsort(discounted_rewards)[-n_elite:]
            elite_actions = actions[elite_idx]

            # Update probabilities based on elite actions
            action_mask = jax.nn.one_hot(elite_actions, num_classes=4)
            new_probs = jnp.sum(action_mask, axis=0) / n_elite

            return (new_probs, key), None

        (final_probs, _), _ = jax.lax.scan(
            iteration, (probs, key), None, length=n_iterations
        )

        # Choose the most probable action for the first step
        return jnp.argmax(final_probs[0])


def main():
    """
    Allows to play with the MordorHike environment using keyboard controls.
    """
    jax.config.update("jax_platform_name", "cpu")

    env = MordorHikeJAX.hard(
        render_mode="human",
        lateral_action="strafe",
        estimate_belief=True,
        num_particles=1000,
    )

    obs, _ = env.reset(seed=20)
    print(f"Initial observation: {obs}")

    cum_rew = 0.0
    done = False

    while not done:
        env.render()
        # cv2.imshow("Mordor Hike", cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("w"):
            action = 0  # Move forward
        elif key == ord("s"):
            action = 1  # Move backward
        elif key == ord("a"):
            action = 2  # Turn left
        elif key == ord("d"):
            action = 3  # Turn right
        else:
            continue
        # Benchmark time per step
        start_time = time.time()
        obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        time_per_step = time.time() - start_time
        print(f"Time per step: {time_per_step}")
        print(f"Action: {action}, Observation: {obs}, Reward: {rew}, Done: {done}")

        cum_rew += rew

    print(f"Total reward: {cum_rew}")
    env.close()
    cv2.destroyAllWindows()


def main_cem():
    # jax.config.update("jax_platform_name", "cpu")

    env = MordorHikeJAX.hard(
        render_mode="human",
        lateral_action="strafe",
        estimate_belief=True,
        num_particles=1000,
    )

    obs, _ = env.reset(seed=20)
    print(f"Initial observation: {obs}")

    cum_rew = 0.0
    done = False

    while not done:
        env.render()

        # Use CEM planning to choose the action
        action = env.cem_plan(
            env.state, env.key, n_iterations=5, n_samples=1000, n_elite=100, horizon=200
        )

        obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        print(f"Action: {action}, Observation: {obs}, Reward: {rew}, Done: {done}")

        cum_rew += rew

        time.sleep(0.1)  # Add a small delay to make the visualization easier to follow

    print(f"Total reward: {cum_rew}")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    main_cem()
