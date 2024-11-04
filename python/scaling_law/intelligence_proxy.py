# %%

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Define the Chinchilla scaling law loss function
# L = E + A/N^alpha + B/D^beta
# These numbers are from Hoffman et al. 2022
alpha = 0.34
beta = 0.28
A = 406.4
B = 410.7
E = 1.69


def loss(n, d) -> float:
    return E + A / n**alpha + B / d**beta


def opt_params(L_budget: float) -> tuple[float, float]:
    l = L_budget - E
    N_opt = (A * (alpha + beta) / (l * beta)) ** (1 / alpha)
    D_opt = (B * (alpha + beta) / (l * alpha)) ** (1 / beta)
    return N_opt, D_opt


base_llm_benchmark_eval = pd.read_csv("./data_models/meta/base_llm_benchmark_eval.csv")

# add PC1- to the datafraome
base_llm_benchmark_eval["PC-1"] = (
    0.45 * base_llm_benchmark_eval["MMLU"] + 
    0.34 * base_llm_benchmark_eval["ARC-C"] + 
    0.38 * base_llm_benchmark_eval["HellaSwag"] + 
    0.24 * base_llm_benchmark_eval["Winograd"] + 
    0.08 * base_llm_benchmark_eval["TruthfulQA"] + 
    0.55 * base_llm_benchmark_eval["GSM8K"] +
    0.21 * base_llm_benchmark_eval["XWinograd"] +
    0.35 * base_llm_benchmark_eval["HumanEval"] 
)

# add optimal params to the dataframe
l_budget = [
    loss(n*1e9, d*1e12)
    for n, d in zip(
        base_llm_benchmark_eval["Model Size (B)"],
        base_llm_benchmark_eval["Pretraining Data Size (T)"],
    )
]
n_opt, d_opt = zip(*[opt_params(l) for l in l_budget])
base_llm_benchmark_eval["N_opt"] = n_opt
base_llm_benchmark_eval["D_opt"] = d_opt
base_llm_benchmark_eval["FLOPs_opt (1E21)"] = 6 * base_llm_benchmark_eval["N_opt"] * base_llm_benchmark_eval["D_opt"]/1e21

fig, ax = plt.subplots(1, 3, figsize=(14, 7)) # 3 columns
# Set the plot labels and title
ax[0].set_title("MMLU vs FLOPs")
ax[0].set_xlabel("log10 FLOPs (1E21)")
ax[0].set_ylabel("MMLU")
ax[0].scatter(np.log10(base_llm_benchmark_eval["FLOPs (1E21)"]), base_llm_benchmark_eval["MMLU"])

ax[1].set_title("MMLU vs FLOPs_opt")
ax[1].set_xlabel("log10 FLOPs_opt (1E21)")
ax[1].set_ylabel("MMLU")
ax[1].scatter(np.log10(base_llm_benchmark_eval["FLOPs_opt (1E21)"]), base_llm_benchmark_eval["MMLU"])

ax[2].set_title("MMLU vs PC-1")
ax[2].set_xlabel("PC-1")
ax[2].set_ylabel("MMLU")
ax[2].scatter(base_llm_benchmark_eval["PC-1"], base_llm_benchmark_eval["MMLU"])

plt.show()