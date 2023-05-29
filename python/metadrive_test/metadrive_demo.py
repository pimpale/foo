import metadrive  # Import this package to register the environment!
import gymnasium as gym

env = gym.make("MetaDrive-validation-v0", config={"render_mode": "human"})
#env = metadrive.MetaDriveEnv(config={"render_mode": "human", "num_scenarios": 100})
env.reset()
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
env.close()
