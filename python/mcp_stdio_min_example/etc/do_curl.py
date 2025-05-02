#!/usr/bin/env python3
import requests

# Function to execute the prompt with streaming response

def execute_prompt(problem_id, model):
    url = "http://127.0.0.1:5000/execute_prompt"
    params = {"problem_id": problem_id, "model": model}
    headers = {
        "Host": "127.0.0.1:5000",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:138.0) Gecko/20100101 Firefox/138.0",
        "Accept": "text/event-stream",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Origin": "http://localhost:5555",
        "Connection": "keep-alive",
        "Referer": "http://localhost:5555/",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Priority": "u=4",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

    # Perform a streaming GET request
    response = requests.get(url, headers=headers, params=params, stream=True)
    response.raise_for_status()

    # Print each line as it arrives
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))


if __name__ == "__main__":
    # Example invocation
    execute_prompt("minexample", "claude-3-7-sonnet-20250219")
