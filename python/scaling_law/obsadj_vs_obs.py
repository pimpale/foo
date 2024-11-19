# %%
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt

from util_linear_obs_scaling_law_predictor import LinearObsScalingLawPredictor
from util_logit_obs_scaling_law_predictor import LogitObsScalingLawPredictor


# Define the Chinchilla loss function parameter set
@dataclass
class ChinchillaParams:
    alpha: float
    beta: float
    A: float
    B: float
    E: float


# These numbers are from Epoch (Besiroglu et al. 2024)
EPOCH_PARAMS = ChinchillaParams(
    alpha=0.3478, beta=0.3658, A=482.01, B=2085.43, E=1.8172
)


def loss(n, d, p: ChinchillaParams) -> float:
    return p.E + p.A / n**p.alpha + p.B / d**p.beta


def opt_params(L_budget: float, p: ChinchillaParams) -> tuple[float, float]:
    l = L_budget - p.E
    N_opt = (p.A * (p.alpha + p.beta) / (l * p.beta)) ** (1 / p.alpha)
    D_opt = (p.B * (p.alpha + p.beta) / (l * p.alpha)) ** (1 / p.beta)
    return N_opt, D_opt


base_llm_benchmark_eval = pd.read_csv("./data_models/meta/base_llm_benchmark_eval.csv")
family_release_dates = duckdb.read_csv("./data_models/meta/family_release_dates.csv")

base_llm_benchmark_eval = duckdb.sql(
    """
    SELECT 
        "Model",
        "Model Family",
        (year(release_date) + (1/365)*dayofyear(release_date)) as release_date,
        "MMLU", 
        "ARC-C", 
        "HellaSwag", 
        "Winograd", 
        "TruthfulQA", 
        "GSM8K", 
        "XWinograd", 
        "HumanEval", 
        "Model Size (B)", 
        "Pretraining Data Size (T)", 
        "FLOPs (1E21)"
    FROM base_llm_benchmark_eval
    JOIN family_release_dates ON base_llm_benchmark_eval."Model Family" = family_release_dates.family
    """
).df()


# add optimal params to the dataframe
for param, label in [(EPOCH_PARAMS, "Besiroglu")]:
    l_budgets = [
        loss(n * 1e9, d * 1e12, param)
        for n, d in zip(
            base_llm_benchmark_eval["Model Size (B)"],
            base_llm_benchmark_eval["Pretraining Data Size (T)"],
        )
    ]
    n_opt, d_opt = zip(*[opt_params(l_budget, param) for l_budget in l_budgets])
    base_llm_benchmark_eval[f"N_opt_{label}"] = n_opt
    base_llm_benchmark_eval[f"D_opt_{label}"] = d_opt
    base_llm_benchmark_eval[f"FLOPs_opt_{label} (1E21)"] = (
        6
        * base_llm_benchmark_eval[f"N_opt_{label}"]
        * base_llm_benchmark_eval[f"D_opt_{label}"]
        / 1e21
    )

# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)

# insert log flops
base_llm_benchmark_eval["log10 FLOPs (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs (1E21)"]
)
base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs_opt_Besiroglu (1E21)"]
)

benchmark_data = [
    ("MMLU", 0.25),
    ("ARC-C", 0.2),
    ("HellaSwag", 0.25),
    ("Winograd", 0.5),
    # ("TruthfulQA", 0.4),
    ("GSM8K", 0.0),
    ("XWinograd", 0.5),
    ("HumanEval", 0.0),
]

benchmarks, benchmark_floor = zip(*benchmark_data)

model_scores = [list(base_llm_benchmark_eval[benchmark]) for benchmark in benchmarks]

logit_obs_model = LogitObsScalingLawPredictor(benchmarks, benchmark_floor, model_scores)
logit_obs_model.fit(optim.Adam(logit_obs_model.parameters(), lr=1e-2))


# %%

fig, ax = plt.subplots(
    len(benchmarks), 4, figsize=(20, len(benchmarks) * 5)
)  # 1 columns
for i, benchmark in enumerate(benchmarks):
    xpoints = np.log10(base_llm_benchmark_eval["FLOPs (1E21)"])
    ypoints = (
        logit_obs_model.predict_benchmark_scores(
            logit_obs_model.predict_benchmark_logit_scores(
                logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
            )
        )
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = logit_obs_model.model_scores.T[i].detach().numpy()
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    ax[i, 0].set_title(f"{benchmark} vs FLOPs")
    ax[i, 0].set_xlabel("log10 FLOPs (1E21)")
    ax[i, 0].set_ylabel(f"{benchmark}")
    ax[i, 0].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 0].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    # ax.plot(xspace, y_sigmoid, color="red")
    ax[i, 0].legend()

    # now plot in flop x-space and logit y-space
    xpoints = np.log10(base_llm_benchmark_eval["FLOPs (1E21)"])
    ypoints = (
        logit_obs_model.predict_benchmark_logit_scores(
            logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
        )
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = logit_obs_model.logit_scores.T[i].detach().numpy()
    ax[i, 1].set_title(f"{benchmark} vs FLOPs")
    ax[i, 1].set_xlabel("log10 FLOPs (1E21)")
    ax[i, 1].set_ylabel(f"{benchmark} logit")
    ax[i, 1].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 1].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    # ax.plot(xspace, y_sigmoid, color="red")
    ax[i, 1].legend()

    # now plot in capability x-space and benchmark y-space
    xpoints = (
        logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
        .detach()
        .numpy()
    )
    ypoints = (
        logit_obs_model.predict_benchmark_scores(
            logit_obs_model.predict_benchmark_logit_scores(
                logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
            )
        )
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = logit_obs_model.model_scores.T[i].detach().numpy()
    ax[i, 2].set_title(f"{benchmark} vs capability")
    ax[i, 2].set_xlabel("Capability")
    ax[i, 2].set_ylabel(f"{benchmark}")
    ax[i, 2].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 2].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    # ax.plot(xspace, y_sigmoid, color="red")
    ax[i, 2].legend()

    # now plot in capability x-space and logit y-space
    xpoints = (
        logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
        .detach()
        .numpy()
    )
    ypoints = (
        logit_obs_model.predict_benchmark_logit_scores(
            logit_obs_model.predict_capability_scores(logit_obs_model.logit_scores)
        )
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = logit_obs_model.logit_scores.T[i].detach().numpy()
    ax[i, 3].set_title(f"{benchmark} vs capability")
    ax[i, 3].set_xlabel("Capability")
    ax[i, 3].set_ylabel(f"{benchmark} logit")
    ax[i, 3].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 3].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    # ax.plot(xspace, y_sigmoid, color="red")
    ax[i, 3].legend()

plt.tight_layout()


# %%
linear_obs_model = LinearObsScalingLawPredictor(benchmarks, model_scores)
linear_obs_model.fit(optim.Adam(linear_obs_model.parameters(), lr=1e-2))

# %%

fig, ax = plt.subplots(
    len(benchmarks), 2, figsize=(10, len(benchmarks) * 5)
)  # 1 columns
for i, benchmark in enumerate(benchmarks):
    xpoints = np.log10(base_llm_benchmark_eval["FLOPs (1E21)"])
    ypoints = (
        linear_obs_model.predict_benchmark_scores(linear_obs_model.predict_capability_scores())
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = linear_obs_model.model_scores.T[i].detach().numpy()
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    ax[i, 0].set_title(f"{benchmark} vs FLOPs")
    ax[i, 0].set_xlabel("log10 FLOPs (1E21)")
    ax[i, 0].set_ylabel(f"{benchmark}")
    ax[i, 0].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 0].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    # ax.plot(xspace, y_sigmoid, color="red")
    ax[i, 0].legend()

    # now plot in capability x-space and benchmark y-space
    xpoints = linear_obs_model.predict_capability_scores().detach().numpy()
    ypoints = (
        linear_obs_model.predict_benchmark_scores(linear_obs_model.predict_capability_scores())
        .T[i]
        .detach()
        .numpy()
    )
    ypoints_2 = linear_obs_model.model_scores.T[i].detach().numpy()
    ax[i, 1].set_title(f"{benchmark} vs capability")
    ax[i, 1].set_xlabel("Capability")
    ax[i, 1].set_ylabel(f"{benchmark}")
    ax[i, 1].scatter(xpoints, ypoints, label=benchmark)
    ax[i, 1].scatter(xpoints, ypoints_2, label=benchmark + " (observed)", alpha=0.5)
    ax[i, 1].legend()


plt.tight_layout()