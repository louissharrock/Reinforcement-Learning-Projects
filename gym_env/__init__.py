from gym.envs.registration import (
    registry,
    register,
    make,
    spec,
    load_env_plugins as _load_env_plugins,
)


register(
    id="GridWorld-v0",
    entry_point="gym_env.envs:GridWorldEnv",
    max_episode_steps=1000,
)
