#%%

import requests
import json
import pathlib

firework_api_key = pathlib.Path("~/tokens/fireworks_mats2024").expanduser().read_text().strip()

url = "https://api.fireworks.ai/inference/v1/completions"
payload = {
  "model": "accounts/fireworks/models/llama-v3p1-405b-instruct",
  "max_tokens": 1,
  "top_p": 1,
  "top_k": 40,
  "logprobs": 0,
  "echo": True,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "temperature": 0.1,
  "prompt": "Hello, how are you?"
}
headers = {
  "Accept": "application/json",
  "Content-Type": "application/json",
  "Authorization": f"Bearer {firework_api_key}"
}
response = requests.request("POST", url, headers=headers, data=json.dumps(payload))


# Parse and print the JSON response
response_data = response.json()
print(json.dumps(response_data, indent=2))