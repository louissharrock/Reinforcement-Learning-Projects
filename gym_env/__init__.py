from gym.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gym_env.envs:GridWorldEnv",
    max_episode_steps=1000,
)
