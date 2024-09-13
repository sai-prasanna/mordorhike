import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from gymnasium import spaces
from scipy.stats import multivariate_normal as mvn


class MordorHike(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    @classmethod
    def easy(cls, **kwargs):
        return cls(occlude_dims=(0, 1), start_distribution="fixed", **kwargs)

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
    ):
        self.dimensions = 2
        self.occluded_dims = list(occlude_dims)

        # dynamics
        self.translate_step = translate_step
        self.rotate_step = rotate_step
        self.translate_std = translate_std
        self.rotate_kappa = rotate_kappa
        self.action_failure_prob = action_failure_prob
        # start state
        self.start_dist = start_distribution

        # belief estimation
        self.estimate_belief = estimate_belief
        self.num_particles = num_particles
        self.effective_particle_threshold = effective_particle_threshold

        # observation
        self.obs_std = obs_std

        # misc
        self.render_mode = render_mode

        self._setup_landscape()

        self.observation_size = 1  # x, y, altitude
        self.action_size = 4  # forward, backward, turn left, turn right

        self.observation_space = spaces.Box(
            low=float("-inf"),
            high=float("inf"),
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.action_size)

        self.window = None

        max_manhattan_distance = np.sum(np.abs(self.upper_bound - self.lower_bound))

        factor = 1
        if self.start_dist == "rotation":
            factor = 2
        elif self.start_dist == "uniform":
            factor = 4
        self.horizon = factor * int(max_manhattan_distance / self.translate_step) * 2

    def _setup_landscape(self):
        self.lower_bound = np.full(self.dimensions, -1.0)
        self.upper_bound = np.full(self.dimensions, 1.0)
        self.fixed_start_pos = np.full(self.dimensions, -0.8)
        self.variable_start_lower_bound = np.array([-1.0, -1.0])
        self.variable_start_upper_bound = np.array([1.0, 0.5])
        self.goal_position = np.full(self.dimensions, 0.8)
        self.mvn_1 = mvn(mean=[0.0, 0.0], cov=[[0.005, 0.0], [0.0, 1.0]])
        self.mvn_2 = mvn(mean=[0.0, -0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
        self.mvn_3 = mvn(mean=[0.0, 0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
        self.slope = np.array([0.2, 0.2])

    def dynamics(self, state, action):
        position, theta = state[..., :2], state[..., 2:]

        success_mask = self.np_random.random(len(theta)) > self.action_failure_prob
        if action < 2:
            forward_vector = (
                np.concatenate(
                    [np.cos(theta[success_mask]), np.sin(theta[success_mask])], axis=-1
                )
                * self.translate_step
            )
            position[success_mask] += forward_vector * (1 - 2 * action)
        elif action >= 2:
            theta[success_mask] += self.rotate_step * (1 - 2 * (action % 2))
            theta[success_mask] = theta[success_mask]

        # Apply Gaussian noise to xy position
        position += self.np_random.normal(0, self.translate_std, position.shape)

        # Apply von Mises noise to rotation
        if self.rotate_kappa is not None:
            theta += self.np_random.vonmises(0, self.rotate_kappa, theta.shape)
        theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

        np.clip(position, self.lower_bound, self.upper_bound, out=position)
        return np.concatenate([position, theta], axis=-1)

    def observation(self, state):
        position = state[..., :2].copy()

        # Calculate altitude
        altitude = self._altitude(position.reshape(-1, 2)).reshape(
            position.shape[:-1] + (1,)
        )

        obs = np.concatenate([position, altitude], axis=-1)

        # Apply observation noise
        obs += self.np_random.normal(0, self.obs_std, obs.shape)

        # Occlude dimensions
        obs = np.delete(obs, self.occluded_dims, axis=-1)

        return obs.astype(np.float32)

    def _altitude(self, position):
        position_2d = position[..., :2]

        mountains = np.stack(
            [
                self.mvn_1.pdf(position_2d),
                self.mvn_2.pdf(position_2d),
                self.mvn_3.pdf(position_2d),
            ]
        )

        altitude = np.max(mountains, axis=0)
        altitude = np.atleast_1d(altitude)

        return (-np.exp(-altitude)) + (position_2d @ self.slope) - 0.02

    def _init_state(self):
        # Calculate the number of possible rotations based on rotate_step
        num_rotations = int(2 * np.pi / self.rotate_step)

        if self.start_dist == "rotation":
            # Pick a random rotation index and calculate theta
            rotation_index = self.np_random.integers(0, num_rotations)
            theta = rotation_index * self.rotate_step
            position = self.fixed_start_pos.copy()
        elif self.start_dist == "uniform":
            # Pick a random rotation index and calculate theta
            rotation_index = self.np_random.integers(0, num_rotations)
            theta = rotation_index * self.rotate_step
            position = self.np_random.uniform(
                self.variable_start_lower_bound,
                self.variable_start_upper_bound,
                self.dimensions,
            )
        elif self.start_dist == "fixed":
            theta = 0
            position = self.fixed_start_pos.copy()
        else:
            raise ValueError(
                f"Invalid start distribution. Must be one of 'rotation', 'uniform', or 'fixed'"
            )

        # Ensure theta is within [0, 2π)
        theta = theta % (2 * np.pi)

        self.state = np.concatenate([position, [theta]])

        if self.estimate_belief:
            self._init_belief()

    def _init_belief(self):
        N = self.num_particles
        if self.start_dist in ["fixed", "rotation"]:
            positions = np.tile(self.fixed_start_pos, (N, 1))
        else:  # uniform
            positions = self.np_random.uniform(
                self.lower_bound, self.upper_bound, (N, self.dimensions)
            )

        num_rotations = int(2 * np.pi / self.rotate_step)
        if self.start_dist == "fixed":
            thetas = np.zeros(N)
        else:
            thetas = self.np_random.integers(0, num_rotations, N) * self.rotate_step

        self.particles = np.column_stack((positions, thetas))
        self.particle_weights = np.ones(N) / N  # Initialize weights

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action. Must be between 0 and {self.action_size - 1}"
            )

        self.state = self.dynamics(self.state.reshape(1, -1), action).reshape(-1)

        observation = self.observation(self.state.reshape(1, -1)).reshape(-1)

        terminated = self._terminal()
        reward = 0 if terminated else self._altitude(self.state[:2].reshape(1, -1))[0]

        truncated = self.step_count >= self.horizon
        info = {}
        if self.estimate_belief:
            self._update_belief(action, observation)
            info["belief"] = self._get_belief()
        self.step_count += 1
        self.path.append(self.state[:2].copy())

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._init_state()
        self.step_count = 0
        self.path = [self.state[:2].copy()]

        observation = self.observation(self.state.reshape(1, -1)).reshape(-1)

        if self.render_mode == "human":
            self.render()

        info = {}
        if self.estimate_belief:
            self._update_belief(None, observation)
            info["belief"] = self._get_belief()
        return observation, info

    def _terminal(self):
        return (
            np.linalg.norm(self.state[:2] - self.goal_position)
            <= self.translate_step * 2
        )

    def _update_belief(self, action, observation):
        if action is not None:
            self.particles = self.dynamics(self.particles, action)

        particle_observations = self.observation(self.particles)

        likelihood = scipy.stats.norm.pdf(
            observation,
            loc=particle_observations,
            scale=self.obs_std,
        ).prod(axis=-1)

        self.particle_weights *= likelihood
        self.particle_weights += (
            1e-300  # Add small constant to prevent division by zero
        )
        self.particle_weights /= np.sum(self.particle_weights)  # Normalize weights

        # Calculate effective sample size
        n_eff = 1 / np.sum(np.square(self.particle_weights))

        # Resample only if effective sample size is below threshold
        if n_eff < self.effective_particle_threshold * self.num_particles:
            indices = self.np_random.choice(
                self.num_particles,
                self.num_particles,
                p=self.particle_weights,
                replace=True,
            )
            self.particles = self.particles[indices]
            self.particle_weights = np.ones(self.num_particles) / self.num_particles

    def _get_belief(self):
        return self.particles.copy(), self.particle_weights.copy()

    def render(self):
        if self.render_mode is None:
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        x = y = np.linspace(self.lower_bound[0], self.upper_bound[0], 100)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)
        Z = self._altitude(positions.reshape(-1, 2)).reshape(X.shape)

        contour = ax.contour(X, Y, Z, levels=20, cmap="viridis")
        ax.clabel(contour, inline=True, fontsize=8)

        path = np.array(self.path)
        ax.plot(path[:, 0], path[:, 1], "r-", linewidth=2, label="Path")

        ax.scatter(
            self.fixed_start_pos[0],
            self.fixed_start_pos[1],
            c="g",
            s=100,
            label="Start",
        )
        ax.scatter(
            self.goal_position[0], self.goal_position[1], c="b", s=100, label="Goal"
        )

        # Render particles with directions and weighted sizes
        particle_sizes = (
            20 + 180 * self.particle_weights
        )  # Scale sizes based on weights
        scatter = ax.scatter(
            self.particles[:, 0],
            self.particles[:, 1],
            c="orange",
            s=particle_sizes,
            alpha=0.6,
            label="Particles",
        )

        # Add particle directions
        ax.quiver(
            self.particles[:, 0],
            self.particles[:, 1],
            np.cos(self.particles[:, 2]),
            np.sin(self.particles[:, 2]),
            color="orange",
            alpha=0.6,
            scale=15,
            width=0.003,
        )

        # Render actual position and direction on top of particles
        ax.scatter(
            self.state[0],
            self.state[1],
            c="red",
            s=150,
            alpha=1.0,
            label="Actual Position",
            zorder=10,
        )
        ax.quiver(
            self.state[0],
            self.state[1],
            np.cos(self.state[2]),
            np.sin(self.state[2]),
            color="red",
            scale=10,
            zorder=11,
        )

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        plt.title("Mordor Hike")
        plt.colorbar(contour, label="Altitude")

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        if self.render_mode == "rgb_array":
            return img
        elif self.render_mode == "human":
            if self.window is None:
                cv2.namedWindow("Mordor Hike", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Mordor Hike", 800, 600)
            cv2.imshow("Mordor Hike", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

    def close(self):
        if self.window is not None:
            cv2.destroyAllWindows()
            self.window = None


def main():
    """
    Allows to play with the MordorHike environment using keyboard controls.
    """
    env = MordorHike.easy(render_mode="human", estimate_belief=True)

    obs, _ = env.reset()
    print(f"Initial observation: {obs}")

    cum_rew = 0.0
    done = False

    while not done:
        env.render()
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

        obs, rew, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        print(f"Action: {action}, Observation: {obs}, Reward: {rew}, Done: {done}")

        cum_rew += rew

    print(f"Total reward: {cum_rew}")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
