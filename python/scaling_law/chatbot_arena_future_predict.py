#%%

import numpy as np
import duckdb
import matplotlib.pyplot as plt
import re

openllm_release_dates = duckdb.read_csv('./data_models/meta/open_llm_release_dates.csv')

model_name_mapping = duckdb.read_csv('./data_models/meta/model_name_mapping.csv')

chatbot_arena_scores = duckdb.read_csv('./data_models/cache/chatbot_arena_20241017.csv')

def extract_a_tag_content(text: str) -> str:
    pattern = r"<a.*>(.*?)</a>"
    match = re.search(pattern, text)
    return match.group(1) if match else None



print(duckdb.sql('DESCRIBE chatbot_arena_scores'))