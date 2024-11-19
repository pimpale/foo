# %%
from dataclasses import dataclass
import duckdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import torch.optim as optim

from util_logit_obs_scaling_law_predictor import LogitObsScalingLawPredictor
import util_plot_markers


# Define the Chinchilla loss function parameter set
@dataclass
class ChinchillaParams:
    alpha: float
    beta: float
    A: float
    B: float
    E: float


# These numbers are from Hoffman et al. 2022
HOFF_PARAMS = ChinchillaParams(
    alpha=0.3392,
    beta=0.2849,
    A=406.4,
    B=410.7,
    E=1.6934,
)

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


############################################
# Calculate Optimal Flops for each model
############################################

# add optimal params to the dataframe
for param, label in [(HOFF_PARAMS, "Hoffman"), (EPOCH_PARAMS, "Besiroglu")]:
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
base_llm_benchmark_eval["log10 FLOPs_Hoffman (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs_opt_Hoffman (1E21)"]
)
base_llm_benchmark_eval["log10 FLOPs_opt_Besiroglu (1E21)"] = np.log10(
    base_llm_benchmark_eval["FLOPs_opt_Besiroglu (1E21)"]
)


############################################
# Calculate Logit Observational Capability Scores
############################################

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

base_llm_benchmark_eval["PC-1"] = logit_obs_model.predict_capability_scores(
    logit_obs_model.logit_scores
).detach().numpy()

#%%
############################################
# Plot each model family on the same graph
# Except, color each line based on time so we can see the progression
############################################
families_release_dates = (
    base_llm_benchmark_eval[["Model Family", "release_date"]].drop_duplicates().values
)

families, release_dates = zip(*families_release_dates)

fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title(f"FLOPs_opt vs PC-1")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("PC-1")
norm = mcolors.Normalize(vmin=min(release_dates), vmax=max(release_dates))
cmap = plt.get_cmap("viridis")
fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, label="Release Date")

linfit_release_dates = []
linfit_yintercepts = []
linfit_slopes = []

for i, (family, release_date) in enumerate(families_release_dates):
    print("release_date", release_date)
    xpoints = np.log10(
        base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
            "FLOPs_opt_Besiroglu (1E21)"
        ]
    )
    ypoints = base_llm_benchmark_eval[
        base_llm_benchmark_eval["Model Family"] == family
    ]["PC-1"]
    # Fit a line to the data
    lparams = np.polyfit(xpoints, ypoints, 1)

    # discard outliers
    if lparams[0] < 0:
        continue

    # add the line fit to the list
    linfit_release_dates.append(release_date)
    linfit_yintercepts.append(lparams[1])
    linfit_slopes.append(lparams[0])

    # plot
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    y_line = np.polyval(lparams, xspace)
    # compute MSE
    linear_mse = np.mean((ypoints - np.polyval(lparams, xpoints)) ** 2)

    ax.scatter(xpoints, ypoints, marker=util_plot_markers.markers[i], label=family)
    ax.plot(xspace, y_line, label="Linear", color=cmap(norm(release_date)))

ax.legend()

fig, ax = plt.subplots(2, 1, figsize=(14, 14))  # 1 columns

ax[0].set_title("Slope of Linear Fit vs Release Date")
ax[0].set_xlabel("Release Date")
ax[0].set_ylabel("Slope")
ax[0].scatter(linfit_release_dates, linfit_slopes)

# fit line
x = np.array(linfit_release_dates)
y = np.array(linfit_slopes)
m, b = np.polyfit(x, y, 1)
ax[0].plot(x, m * x + b, label=f"y = {m:.2f}x + {b:.2f}")
ax[0].legend()


ax[1].set_title("Y-intercept of Linear Fit vs Release Date")
ax[1].set_xlabel("Release Date")
ax[1].set_ylabel("Y-intercept")
ax[1].scatter(linfit_release_dates, linfit_yintercepts)

# fit line
x = np.array(linfit_release_dates)
y = np.array(linfit_yintercepts)
m, b = np.polyfit(x, y, 1)
ax[1].plot(x, m * x + b, label=f"y = {m:.2f}x + {b:.2f}")
ax[1].legend()

# %%
