import gymnasium as gym
from gymnasium.envs.registration import WrapperSpec
from powm.envs.mordor import MordorHike

gym.register(
    "mordor-hike-v0",
    entry_point="powm.envs.mordor:MordorHike",
    additional_wrappers=(
        WrapperSpec(
            "render-wrapper",
            "powm.envs.wrappers:RenderWrapper",
            kwargs={"render_key": "log_image", "obs_key": "vector"},
        ),
    ),
)
for difficulty in ["easy", "medium", "hard"]:
    gym.register(
        f"mordor-hike-{difficulty}-v0",
        entry_point=getattr(MordorHike, difficulty),
        additional_wrappers=(
            WrapperSpec(
                "render-wrapper",
                "powm.envs.wrappers:RenderWrapper",
                kwargs={"render_key": "log_image", "obs_key": "vector"},
            ),
        ),
    )