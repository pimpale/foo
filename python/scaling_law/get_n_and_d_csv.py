# %%
import duckdb
import pandas as pd
import numpy as np

# model,params,tokens,explanation
# qwen2_5_72b, 72e9, 18e12, 6 * (72*10^9)*(18*10^12)
# qwen2_5_32b, 32e9, 18e12, 6 * (32*10^9)*(18*10^12)
# qwen2_5_14b, 14e9, 18e12, 6 * (14*10^9)*(18*10^12)
# qwen2_5_7b, 7e9, 18e12, 6 * (7*10^9)*(18*10^12)
# qwen2_5_3b, 3e9, 18e12, 6 * (3*10^9)*(18*10^12)
# qwen2_5_1_5b, 1.5e9, 18e12, 6 * (1.5*10^9)*(18*10^12)
# qwen2_5_0_5b, 0.5e9, 18e12, 6 * (0.5*10^9)*(18*10^12)
# llama3_2_3b, 3e9, 9e12, 6 * (3*10^9)*(9*10^12)
# llama3_2_1b, 1e9, 9e12, 6 * (1*10^9)*(9*10^12)
# phi3_mini_128k, 3.8e9, 4.9e12, 6 * (3.8*10^9) * (4.9*10^12) see https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
# phi3_small_128k, 7e9, 4.9e12, 6 * (7*10^9) * (4.9*10^12) see https://huggingface.co/microsoft/Phi-3-small-128k-instruct
# phi3_medium_128k, 14e9, 4.9e12, 6 * (14*10^9) * (4.9*10^12) see https://huggingface.co/microsoft/Phi-3-medium-128k-instruct

base_llm_benchmark_eval = duckdb.read_csv("./data_models/meta/base_llm_benchmark_eval.csv")

model_n_and_d = duckdb.sql(
    """
    SELECT Model as model,"Model Family" as family,"Model Size (B)" as N, "Pretraining Data Size (T)" as D FROM base_llm_benchmark_eval
    UNION ALL VALUES
        ('qwen2_5_72b', 'qwen2', 72, 18),
        ('qwen2_5_32b', 'qwen2', 32, 18),
        ('qwen2_5_14b', 'qwen2', 14, 18),
        ('qwen2_5_7b', 'qwen2', 7, 18),
        ('qwen2_5_3b', 'qwen2', 3, 18),
        ('qwen2_5_1_5b', 'qwen2', 1.5, 18),
        ('qwen2_5_0_5b', 'qwen2', 0.5, 18),
        ('llama3_2_3b', 'llama3.2', 3, 9),
        ('llama3_2_1b', 'llama3.2', 1, 9),
        ('phi3_mini_128k', 'phi3', 3.8, 4.9),
        ('phi3_small_128k', 'phi3', 7, 4.9),
        ('phi3_medium_128k', 'phi3', 14, 4.9);
    
    """
).df()

# write to csv
model_n_and_d.to_csv("./data_models/meta/compute_estimates_nd.csv", index=False)