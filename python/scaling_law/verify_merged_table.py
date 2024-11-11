# %%

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read merged.csv
merged = duckdb.read_csv("./data_models/test/merged.csv")

merged_df = merged.to_df()

merged_df.dropna(
    inplace=True,
    subset=[
        "chatbot_arena_name",
        "arena_score",
        # "release_date",
        "open_llm_name",
        "model",
        "IFEval Raw",
        "BBH Raw",
        "MATH Lvl 5 Raw",
        "GPQA Raw",
        "MUSR Raw",
        "MMLU-PRO Raw",
    ],
)
