import metadrive  # Import this package to register the environment!
import gymnasium as gym

def print_keys():
    for key in gym.registry:
        print(key)
        print(gym.registry[key])

env = gym.make("MetaDrive-validation-v0", config=dict(render_mode="human"))
env.reset()
for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    print(obs)
    env.render()
    if terminated or truncated:
        env.reset()
env.close()
