keys = {
    "key1": {
        "backoff": 1,
        "ok_at": datetime.now(),
        "max_ok_threads"
    }
}

def do_run():
    key = random.choice(list(keys.keys()))
    messages = []
    while True:
        try:
            if datetime.now() < keys[key]["ok_at"]:
                time.sleep(keys[key]["ok_at"] - datetime.now())
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
        except Exception429:
            keys[key]["backoff"] += 1
            keys[key]["ok_at"] = datetime.now() + 2**keys[key]["backoff"] + random.random()

            


if __name__ == "__main__":
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(do_run) for _ in range(10)]
        for future in concurrent.futures.as_completed(futures):
            future.result()