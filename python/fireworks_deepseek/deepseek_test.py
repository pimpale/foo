#%%

import requests
import json
import pathlib

deepseek_api_key = pathlib.Path("~/tokens/deepseek_mats_2025").expanduser().read_text().strip()

url = "https://api.deepseek.com/beta/completions"
payload = {
  "model": "deepseek-chat",
  "max_tokens": 100,
  "top_p": 1,
  "top_k": 40,
#   "logprobs": 0,
  "echo": True,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.1,
  "prompt": "Hello, how are you?"
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {deepseek_api_key}"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))


# Parse and print the JSON response
response_data = response.json()
print(json.dumps(response_data, indent=2))


# %%
# try deepseek reasoning model

url = "https://api.deepseek.com/beta/completions"
payload = {
  "model": "deepseek-reasoning",
  "max_tokens": 100,
  "top_p": 1,
  "top_k": 40,
#   "logprobs": 0,
  "echo": True,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.1,
  "prompt": "Hello, how are you?"
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {deepseek_api_key}"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))