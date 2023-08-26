import metadrive  # Import this package to register the environment!
import gymnasium as gym

env = gym.make("MetaDrive-validation-v0", config={"use_render": True})
env.reset()

for _ in range(10):
    env.step([0,0])

env.vehicle.set_velocity([0, 0], in_local_frame=True)
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step([0, 1])
    if terminated or truncated:
        env.reset()
env.close()
