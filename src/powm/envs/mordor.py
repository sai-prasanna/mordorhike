import random
import time
import warnings

import cv2
import gymnasium as gym
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

    @classmethod
    def veryhard(cls, **kwargs):
        return cls(
            occlude_dims=(0, 1),
            start_distribution="rotation",
            lateral_action="rotate",
            **kwargs,
        )

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
        render_size=(128, 128),
        lateral_action="strafe",
    ):
        self.dimensions = 2
        self.occluded_dims = list(occlude_dims)

        # dynamics
        self.translate_step = translate_step
        self.rotate_step = rotate_step if lateral_action == "rotate" else np.pi / 2
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
        self.render_size = render_size

        # action mode
        self.lateral_action = lateral_action

        self._setup_landscape()

        self.observation_size = 1  # x, y, altitude
        self.action_size = (
            4  # north, south, east, west or forward, backward, turn left, turn right
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_size,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.action_size)

        self.window = None

        max_manhattan_distance = np.sum(
            np.abs(self.map_upper_bound - self.map_lower_bound)
        )

        factor = 1
        if self.start_dist == "rotation":
            factor = 2
        elif self.start_dist == "uniform":
            factor = 4
        self.horizon = factor * int(max_manhattan_distance / self.translate_step) * 2

    def _setup_landscape(self):
        self.map_lower_bound = np.full(self.dimensions, -1.0)
        self.map_upper_bound = np.full(self.dimensions, 1.0)
        self.fixed_start_pos = np.full(self.dimensions, -0.8)
        self.uniform_start_lower_bound = np.array([-1.0, -1.0])
        self.uniform_start_upper_bound = np.array([0.0, 0.0])
        self.goal_position = np.full(self.dimensions, 0.8)
        self.mvn_1 = mvn(mean=[0.0, 0.0], cov=[[0.005, 0.0], [0.0, 1.0]])
        self.mvn_2 = mvn(mean=[0.0, -0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
        self.mvn_3 = mvn(mean=[0.0, 0.8], cov=[[1.0, 0.0], [0.0, 0.01]])
        self.slope = np.array([0.2, 0.2])

    def dynamics(self, state, action):
        # Vectorized action handling
        position, theta = state[..., :2], state[..., 2:]
        batch_size = len(theta)
        success_mask = self.np_random.random(batch_size) > self.action_failure_prob

        # Create action vectors
        forward_action = np.array([1, -1, 0, 0])
        lateral_action = np.array([0, 0, 1, -1])

        # Compute movement vectors
        forward_vector = (
            np.concatenate([np.cos(theta), np.sin(theta)], axis=-1)
            * self.translate_step
        )

        # Compute position updates
        position_update = forward_vector * forward_action[action]

        # Apply updates only to successful actions
        position[success_mask] += position_update[success_mask]

        # Handle rotation if lateral action is "rotate"
        if self.lateral_action == "rotate":
            rotation_update = self.rotate_step * lateral_action[action]
            theta[success_mask] += rotation_update[success_mask]
        else:
            side_vector = (
                np.concatenate([-np.sin(theta), np.cos(theta)], axis=-1)
                * self.translate_step
            )
            position += side_vector * lateral_action[action]

        # Apply Gaussian noise to xy position
        position += self.np_random.normal(0, self.translate_std, position.shape)

        # Apply von Mises noise to rotation
        if self.rotate_kappa is not None:
            theta += self.np_random.vonmises(0, self.rotate_kappa, theta.shape)
        theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

        np.clip(position, self.map_lower_bound, self.map_upper_bound, out=position)
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

        # Clip observation to [-1, 1]
        # np.clip(obs, -1.0, 1.0, out=obs)

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
                self.uniform_start_lower_bound,
                self.uniform_start_upper_bound,
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
                self.uniform_start_lower_bound, self.uniform_start_upper_bound, (N, self.dimensions)
            )

        num_rotations = int(2 * np.pi / self.rotate_step)
        if self.start_dist == "fixed":
            thetas = np.zeros(N)
        else:
            thetas = self.np_random.integers(0, num_rotations, N) * self.rotate_step

        self.particles = np.column_stack((positions, thetas))

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action. Must be between 0 and {self.action_size - 1}"
            )

        self.state = self.dynamics(
            self.state.reshape(1, -1), np.array([action])
        ).reshape(-1)
        observation = self.observation(self.state.reshape(1, -1)).reshape(-1)
        terminated = self._terminal(self.state.reshape(1, -1))[0]
        reward = 0 if terminated else self._altitude(self.state[:2].reshape(1, -1))[0]

        truncated = self.step_count >= self.horizon
        info = {
            "state": self.state.copy()
        }
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

        info = {
            "state": self.state.copy()
        }
        if self.estimate_belief:
            self._init_belief()
            self._update_belief(None, observation)
            info["belief"] = self._get_belief()
        return observation, info

    def _terminal(self, states):
        return (
            np.linalg.norm(states[:, :2] - self.goal_position, axis=-1)
            <= self.translate_step * 2
        )

    def _update_belief(self, action, observation):
        if action is not None:
            self.particles = self.dynamics(
                self.particles, np.tile(action, (self.num_particles, 1))
            )

        particle_observations = self.observation(self.particles)

        likelihood = scipy.stats.norm.pdf(
            observation,
            loc=particle_observations,
            scale=self.obs_std,
        ).prod(axis=-1)

        # Update weights
        weights = likelihood
        weights /= np.sum(weights)  # Normalize weights

        # Always resample
        indices = self.np_random.choice(
            self.num_particles,
            self.num_particles,
            p=weights,
            replace=True,
        )
        self.particles = self.particles[indices]

    def _get_belief(self):
        return self.particles

    def render(self):
        if self.render_mode is None:
            return

        # Create or get cached background
        if not hasattr(self, "background"):
            self.background = self._create_background()

        # Create a copy of the background to draw on
        img = self.background.copy()

        # Draw path
        if len(self.path) > 1:
            path = self._world_to_pixel(self.path).astype(np.int32)
            cv2.polylines(img, [path], False, (0, 0, 255), 2)

        scale_factor = int(max(self.render_size) / 128)
        # Draw start and goal
        # start = tuple(map(int, self._world_to_pixel(self.fixed_start_pos)))
        goal = tuple(map(int, self._world_to_pixel(self.goal_position)))

        # cv2.circle(img, start, int(4 * scale_factor), (0, 255, 0), -1)
        cv2.circle(img, goal, int(4 * scale_factor), (255, 0, 0), -1)

        # Draw particles
        if self.estimate_belief:
            # Create density map for particle positions
            density_map = np.zeros(img.shape[:2], dtype=np.float32)
            for particle in self.particles:
                pos = tuple(map(int, self._world_to_pixel(particle[:2])))
                if 0 <= pos[0] < img.shape[1] and 0 <= pos[1] < img.shape[0]:
                    density_map[pos[1], pos[0]] += 1
            
            # Normalize density map
            density_map = density_map / np.max(density_map)
            
            # Draw particles with colors based on angle difference
            for particle in self.particles:
                pos = tuple(map(int, self._world_to_pixel(particle[:2])))
                if 0 <= pos[0] < img.shape[1] and 0 <= pos[1] < img.shape[0]:
                    # Calculate angle difference with true state
                    angle_diff = abs((particle[2] - self.state[2] + np.pi) % (2*np.pi) - np.pi)
                    angle_match = angle_diff < np.pi/6  # Within 30 degrees
                    
                    # Set alpha based on local density
                    alpha = min(1.0, density_map[pos[1], pos[0]] * 0.8 + 0.2)
                    
                    # Draw circle and direction with different colors based on angle match
                    color = (0, 255, 0) if angle_match else (0, 165, 255)  # Green if angle matches, orange if not
                    overlay = img.copy()
                    cv2.circle(overlay, pos, int(2), color, 1)
                    
                    direction = (
                        int(4 * scale_factor * np.cos(particle[2])),
                        int(-4 * scale_factor * np.sin(particle[2])),
                    )
                    cv2.line(
                        overlay,
                        pos,
                        (pos[0] + direction[0], pos[1] + direction[1]),
                        color,
                        1,
                    )
                    
                    # Apply alpha blending
                    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # Draw actual position and direction
        pos = tuple(map(int, self._world_to_pixel(self.state[:2])))
        cv2.circle(img, pos, int(5 * scale_factor), (0, 0, 255), -1)
        direction = (
            int(7 * scale_factor * np.cos(self.state[2])),
            int(-7 * scale_factor * np.sin(self.state[2])),
        )
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
        x = np.linspace(self.map_lower_bound[0], self.map_upper_bound[0], width)
        y = np.linspace(self.map_upper_bound[1], self.map_lower_bound[1], height)
        X, Y = np.meshgrid(x, y)
        positions = np.stack([X, Y], axis=-1)
        Z = self._altitude(positions.reshape(-1, 2)).reshape(height, width)

        # Normalize Z to 0-255 range
        Z_norm = ((Z - Z.min()) / (Z.max() - Z.min()) * 255).astype(np.uint8)

        # Create color map
        color_map = cv2.applyColorMap(Z_norm, cv2.COLORMAP_VIRIDIS)

        # Draw contours
        levels = np.linspace(Z.min(), Z.max(), 10)
        for level in levels:
            contours = cv2.findContours(
                (Z >= level).astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )[0]
            cv2.drawContours(color_map, contours, -1, (255, 255, 255), 1)

        return color_map

    def _world_to_pixel(self, coord):
        coord = np.asarray(coord)
        x = np.floor(
            (coord[..., 0] - self.map_lower_bound[0])
            / (self.map_upper_bound[0] - self.map_lower_bound[0])
            * (self.render_size[1] - 1)
        ).astype(np.int32)
        # Flip y-axis for image coordinates
        y = np.floor(
            (
                1
                - (coord[..., 1] - self.map_lower_bound[1])
                / (self.map_upper_bound[1] - self.map_lower_bound[1])
            )
            * (self.render_size[0] - 1)
        ).astype(np.int32)
        return np.stack([x, y], axis=-1)

    def cem_plan(
        self,
        state,
        n_iterations=5,
        n_samples=1000,
        n_elite=100,
        horizon=50,
        discount=0.99,
        update_smooth_factor=0.1,
    ):
        # Initialize probabilities for each action (4 actions)
        probs = np.ones((horizon, 4)) / 4
        state = np.tile(state, (n_samples, 1))

        for _ in range(n_iterations):
            # Sample actions from multinomial distribution for each time step
            actions = np.array(
                [
                    self.np_random.choice(4, size=n_samples, p=probs[t])
                    for t in range(horizon)
                ]
            ).T

            actions = actions[..., np.newaxis]
            returns = self._rollout(state, actions, discount)

            # Select elite samples
            elite_idx = np.argsort(returns)[-n_elite:]
            elite_actions = actions[elite_idx]

            # Update probabilities based on elite actions
            new_probs = np.eye(4)[elite_actions].squeeze().mean(axis=0)
            probs = (
                update_smooth_factor * new_probs + (1 - update_smooth_factor) * probs
            )

        # Choose the most probable action for the first step
        return np.argmax(probs[0])

    def _rollout(self, state, actions, discount):
        rewards = np.zeros((len(state), actions.shape[1]))

        for t in range(actions.shape[1]):
            state = self.dynamics(state, actions[:, t])
            rewards[:, t] = self._altitude(state[:, :2])
            terminated = self._terminal(state)
            rewards[terminated, t:] = 0
            if np.all(terminated):
                break

        discount_factors = discount ** np.arange(actions.shape[1])
        returns = (rewards * discount_factors).sum(axis=1)
        return returns

    def reinforce_plan(
        self,
        state,
        n_iterations=1000,
        horizon=50,
        learning_rate=0.01,
        discount=0.99,
        batch_size=32,
    ):
        # Initialize action probabilities
        log_probs = np.ones((horizon, 4)) / 4

        for _ in range(n_iterations):
            # Sample actions
            actions = np.array(
                [
                    self.np_random.choice(
                        4, p=np.exp(log_probs[t]) / np.sum(np.exp(log_probs[t]))
                    )
                    for t in range(horizon)
                ]
            )

            # Perform rollouts
            returns = np.zeros((batch_size, horizon))
            for b in range(batch_size):
                current_state = state.copy()
                for t in range(horizon):
                    current_state = self.dynamics(
                        current_state.reshape(1, -1), np.array([actions[t]])
                    ).reshape(-1)
                    reward = (
                        0
                        if self._terminal(current_state.reshape(1, -1))[0]
                        else self._altitude(current_state[:2].reshape(1, -1))[0]
                    )
                    returns[b, t:] += reward * (discount ** np.arange(horizon - t))
                    if self._terminal(current_state.reshape(1, -1))[0]:
                        break

            # Compute gradient and update log_probs
            for t in range(horizon):
                action_mask = np.eye(4)[actions[t]]
                baseline = np.mean(returns[:, t])
                advantage = returns[:, t] - baseline
                grad = np.mean(
                    action_mask[np.newaxis, :] * advantage[:, np.newaxis], axis=0
                )
                log_probs[t] += learning_rate * grad

        # Choose the most probable action for the first step
        return np.argmax(log_probs[0])
        
    def discretize_belief(self, belief, grid_x_y_size=0.1, grid_angle_size=np.pi/2):
        n_bins_x = np.ceil((self.map_upper_bound[0] - self.map_lower_bound[0]) / grid_x_y_size).astype(int)
        n_bins_y = np.ceil((self.map_upper_bound[1] - self.map_lower_bound[1]) / grid_x_y_size).astype(int)
        n_bins_angle = np.ceil(2*np.pi / grid_angle_size).astype(int)
        # Use numpy's histogram2d for better binning
        grid, _ = np.histogramdd(
            belief[:, :3],  # x, y, angle coordinates
            bins=[n_bins_x, n_bins_y, n_bins_angle],
            range=[
                [self.map_lower_bound[0], self.map_upper_bound[0]],
                [self.map_lower_bound[1], self.map_upper_bound[1]],
                [0, 2*np.pi]
            ],
            density=True  # This automatically normalizes the histogram
        )
        return grid / grid.sum()

    def get_waypoint_action(self, waypoints, visited=None):
        """Returns action to move towards next unvisited waypoint.
        
        Args:
            waypoints: List of (x,y) waypoint positions
            visited: List of booleans indicating which waypoints have been visited.
                    If None, initializes new visited array.
        
        Returns:
            Tuple of (action, visited), where action is the next action to take
            (or None if all waypoints visited) and visited is the updated visited array
        """
        # Initialize visited array if None
        if visited is None:
            visited = np.zeros(len(waypoints), dtype=bool)
        
        # Get current position
        current_pos = self.state[:2]
        
        # Check and update if we've reached current waypoint
        for i, (waypoint, is_visited) in enumerate(zip(waypoints, visited)):
            if not is_visited and np.linalg.norm(current_pos - waypoint) <= self.translate_step * 2:
                visited[i] = True
        
        # Find next unvisited waypoint
        next_waypoint = None
        for i, (waypoint, is_visited) in enumerate(zip(waypoints, visited)):
            if not is_visited:
                next_waypoint = waypoint
                break
        
        if next_waypoint is None:
            return None, visited  # All waypoints visited
        
        # Get current orientation
        theta = self.state[2]
        
        # Calculate direction vector to waypoint
        direction = next_waypoint - current_pos
        
        # Calculate angle between forward vector and goal direction
        angle = np.arctan2(direction[1], direction[0]) - theta
        angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
        
        is_rotate = self.lateral_action == "rotate"
        
        if is_rotate:
            # If roughly facing goal, move forward
            if abs(angle) < np.pi/4:
                return 0, visited  # Forward
            # Otherwise rotate towards goal
            elif angle > 0:
                return 2, visited  # Turn left
            else:
                return 3, visited  # Turn right
        else:
            # For lateral movement, we can move sideways
            if abs(angle) < np.pi/4:
                return 0, visited  # Forward
            elif abs(angle - np.pi/2) < np.pi/4:
                return 2, visited  # Strafe left
            elif abs(angle + np.pi/2) < np.pi/4:
                return 3, visited  # Strafe right
            else:
                return 1, visited  # Backward

    def generate_random_waypoints(self, n_waypoints, rng, final_waypoint_at_center=True):
        """Generate random waypoints within the map bounds.
        
        Args:
            n_waypoints: Number of waypoints to generate
            rng: Random number generator instance
            final_waypoint_at_center: If True, makes the last waypoint at map center
            
        Returns:
            Array of shape (n_waypoints, 2) containing waypoint positions
        """
        # Calculate map center
        map_center = (self.map_upper_bound + self.map_lower_bound) / 2
        
        # Generate initial random points
        waypoints = []
        # Minimum distance between waypoints and waypoint from goal
        min_dist = self.translate_step * 10  
        goal = self.goal_position
        
        # Adjust target number of waypoints if we're adding center point
        target_waypoints = n_waypoints - 1 if final_waypoint_at_center else n_waypoints
        
        # Keep trying until we have enough valid waypoints
        attempts = 0
        while len(waypoints) < target_waypoints and attempts < 1000:
            # Generate random point in [-1, 1] range
            point = rng.uniform(self.map_lower_bound, self.map_upper_bound, 2)
            
            # Check if point is far enough from goal
            if np.linalg.norm(point - goal) < min_dist:
                attempts += 1
                continue
                
            # Check if point is too close to previous waypoint
            if len(waypoints) > 0:
                if np.linalg.norm(point - waypoints[-1]) < min_dist:
                    attempts += 1
                    continue
                    
            # If this is the second-to-last point and we're adding center point,
            # ensure it's not too close to center
            if final_waypoint_at_center and len(waypoints) == target_waypoints - 1:
                if np.linalg.norm(point - map_center) < min_dist:
                    attempts += 1
                    continue
                
            waypoints.append(point)
            attempts += 1
        
        if len(waypoints) < target_waypoints:
            warnings.warn(f"Could only generate {len(waypoints)} waypoints")
        
        # Add center point if requested
        if final_waypoint_at_center:
            waypoints.append(map_center)
        
        return np.array(waypoints)

def main():
    """
    Allows to play with the MordorHike environment using keyboard controls.
    """
    env = MordorHike.hard(
        render_mode="human", lateral_action="strafe", estimate_belief=True
    )
    obs, _ = env.reset(seed=2)

    cum_rew = 0.0
    
    waypoints = np.array([[0.0, 0.0], [-1.0, -1.0]])
    visited = np.zeros(len(waypoints), dtype=bool)
    
    while True:
        done = False
        obs, _ = env.reset()
        while not done:
            start_time = time.time()
            env.render()
            end_time = time.time()
            print(f"Time taken render: {end_time - start_time}")
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
            elif key == ord("c"):
                action = env.cem_plan(
                    env.state,
                    n_iterations=5,
                    n_samples=1000,
                    n_elite=100,
                    horizon=50,
                    discount=0.99,
                    update_smooth_factor=0.1,
                )
            elif key == ord("v"):
                action, visited = env.get_waypoint_action(waypoints, env.state, visited)
                if action is None:
                    print("All waypoints visited or current waypoint reached")
                    action = 0
            elif key == ord("e"):
                sampled_particle = random.choice(env.particles)
                env.particles = np.tile(sampled_particle, (env.num_particles, 1))
                print(sampled_particle, env.particles.shape)
                action = env.cem_plan(
                    sampled_particle,
                    n_iterations=5,
                    n_samples=1000,
                    n_elite=100,
                    horizon=50,
                    discount=0.99,
                    update_smooth_factor=0.1,
                )
            elif key == ord("r"):
                action = env.reinforce_plan(
                    env.state,
                    n_iterations=1000,
                    horizon=50,
                    learning_rate=0.01,
                    discount=0.99,
                    batch_size=32,
                )
            elif key == ord("p"):
                action = env.predefined_policy(mode="diagonal")
            elif key == ord("u"):
                action = env.predefined_policy(mode="up_right")
            elif key == ord("i"):
                action = env.predefined_policy(mode="right_up")
            else:
                continue
            start_time = time.time()
            obs, rew, terminated, truncated, _ = env.step(action)
            end_time = time.time()
            print(f"Time taken: {end_time - start_time}")
            done = terminated or truncated
            print(f"Action: {action}, Observation: {obs}, Reward: {rew}, Done: {done}")
            

            cum_rew += rew

    print(f"Total reward: {cum_rew}")
    env.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
