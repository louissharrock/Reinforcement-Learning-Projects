from gym.envs.registration import register

register(
    id="GridWorld-v0",
    entry_point="gym_gridworld.envs:GridWorldEnv",
    max_episode_steps=1000,
)
