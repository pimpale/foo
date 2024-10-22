# %%
from gradio_client import Client
import pandas as pd
import re

client = Client("lmsys/chatbot-arena-leaderboard")
result_tuple = client.predict(
    category="Overall", filters=[], api_name="/update_leaderboard_and_plots"
)

result = result_tuple[0]['value']

df = pd.DataFrame(result["data"], columns=result["headers"])

def extract_a_tag_content(text: str) -> str|None:
    pattern = r"<a.*>(.*?)</a>"
    match = re.search(pattern, text)
    return match.group(1) if match else None

# extract <a/> tag content
df["Model"] = df["Model"].apply(extract_a_tag_content)

df.to_csv("./data_models/cache_new/chatbot_arena.csv", index=False)