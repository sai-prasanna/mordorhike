import gymnasium as gym
import numpy as np


class RenderWrapper(gym.ObservationWrapper):
    """ Used for passing rendered images for logging in R2I and Dreamer."""
    def __init__(self, env, render_key="image", obs_key="obs", size=(128, 128)):
        super().__init__(env)
        self.render_key = render_key
        self.obs_key = obs_key
        self.size = size
        # Update observation space
        render_shape = (*size, 3)
        if isinstance(self.observation_space, gym.spaces.Dict):
            self.observation_space.spaces[render_key] = gym.spaces.Box(
                0, 255, render_shape, dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    self.obs_key: self.observation_space,
                    render_key: gym.spaces.Box(
                        0, 255, render_shape, dtype=np.uint8
                    ),
                }
            )

    def observation(self, obs):
        rendered_img = self.env.render()
        assert rendered_img.shape == (*self.size, 3)
        if isinstance(obs, dict):
            obs[self.render_key] = rendered_img
        else:
            obs = {self.obs_key: obs, self.render_key: rendered_img}
        return obs