# %%
import browser_env


if __name__ == "__main__":

    env = browser_env.ScriptBrowserEnv(
        headless=False,
        observation_type="accessibility_tree",
        current_viewport_only=True,
    )
    # %%
    env.reset()
    observation, reward, terminated, truncated, info = env.step(browser_env.create_goto_url_action("https://www.wikipedia.com"))

    # %%
    
    while True:
        print(observation['text'])
        print('===============================\n')
        command = input("Enter command: ")
        observation, reward, terminated, truncated, info = env.step(browser_env.create_id_based_action(command))

# %%
