#%%
from gradio_client import Client
import pandas as pd
from pathlib import Path

client = Client("open-llm-leaderboard/open_llm_leaderboard")
result = client.predict(
		api_name="/get_latest_data_leaderboard"
)

df = pd.DataFrame(result["data"], columns=result["headers"])

# drop column 'T' and 'Model'
df.drop(columns=['T', 'Model'], inplace=True)

df.to_csv(Path(__file__).parent / "../data_models/cache_new/open_llm_leaderboard.csv", index=False)