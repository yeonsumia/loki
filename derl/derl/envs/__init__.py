from gym.envs.registration import register

try:
    register(
        id="Unimal-v0",
        entry_point="derl.envs.tasks.task:make_env",
        max_episode_steps=1000,
    )
    pass
except:
    pass # metamorph also register Unimal-v0