# %%
from gradio_client import Client
import pandas as pd
import re
from pathlib import Path

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

def extract_huggingface_name(text: str):
    pattern = r'href="https://huggingface\.co/(.*?)"'
    match = re.search(pattern, text)
    return match.group(1) if match else None


# extract <a/> tag content
df["Huggingface Name"] = df["Model"].apply(extract_huggingface_name)
df["Model"] = df["Model"].apply(extract_a_tag_content)


df.to_csv(Path(__file__).parent / "./data_models/cache_new/chatbot_arena.csv", index=False)