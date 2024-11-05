# %%

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from dataclasses import dataclass
import utils.plot_markers


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

# add PC1- to the dataframe
base_llm_benchmark_eval["PC-1"] = (
    0.45 * base_llm_benchmark_eval["MMLU"]
    + 0.34 * base_llm_benchmark_eval["ARC-C"]
    + 0.38 * base_llm_benchmark_eval["HellaSwag"]
    + 0.24 * base_llm_benchmark_eval["Winograd"]
    + 0.08 * base_llm_benchmark_eval["TruthfulQA"]
    + 0.55 * base_llm_benchmark_eval["GSM8K"]
    + 0.21 * base_llm_benchmark_eval["XWinograd"]
    + 0.35 * base_llm_benchmark_eval["HumanEval"]
)

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


def sigmoid(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - shift)))


def scaled_sigmoid(
    x: np.ndarray, slope: float, shift: float, scale: float, yoffset: float
) -> np.ndarray:
    return scale * sigmoid(x, slope, shift) + yoffset


def get_sigmoid_parameters(
    x_values: np.ndarray,
    y_values: np.ndarray,
    p0: tuple[
        # slope
        float,
        # shift
        float,
    ],
) -> tuple[float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        sigmoid, x_values, y_values, p0=p0, bounds=([0, -4], [10, 4]), maxfev=5000
    )
    return popt


def get_scaled_sigmoid_parameters(
    x_values: np.ndarray,
    y_values: np.ndarray,
    p0: tuple[
        # slope
        float,
        # shift
        float,
        # scale
        float,
        # yoffset
        float,
    ],
) -> tuple[float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        scaled_sigmoid,
        x_values,
        y_values,
        p0=p0,
        bounds=([0, -4, 0, 0], [10, 4, 3, 1]),
        maxfev=5000,
    )
    return popt


# %%
fig, ax = plt.subplots(1, 3, figsize=(21, 7))  # 3 columns
# Set the plot labels and title
ax[0].set_title("MMLU vs FLOPs")
ax[0].set_xlabel("log10 FLOPs (1E21)")
ax[0].set_ylabel("MMLU")
ax[0].scatter(
    np.log10(base_llm_benchmark_eval["FLOPs (1E21)"]), base_llm_benchmark_eval["MMLU"]
)

ax[1].set_title("MMLU vs FLOPs_opt (Besiroglu)")
ax[1].set_xlabel("log10 FLOPs_opt (1E21)")
ax[1].set_ylabel("MMLU")
ax[1].scatter(
    np.log10(base_llm_benchmark_eval["FLOPs_opt_Besiroglu (1E21)"]),
    base_llm_benchmark_eval["MMLU"],
)

ax[2].set_title("MMLU vs PC-1")
ax[2].set_xlabel("PC-1")
ax[2].set_ylabel("MMLU")
ax[2].scatter(base_llm_benchmark_eval["PC-1"], base_llm_benchmark_eval["MMLU"])
plt.show()

# %%
fig, ax = plt.subplots(1, 3, figsize=(21, 7))  # 3 columns
# Set the plot labels and title

xpoints = np.log10(base_llm_benchmark_eval["FLOPs (1E21)"])
ypoints = base_llm_benchmark_eval["PC-1"]

# Fit a sigmoid to the data
params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0=(1, 0, 0, 0.555))
# plot
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
y_sigmoid = scaled_sigmoid(xspace, *params)

# compute MSE
sigmoid_mse = np.mean((ypoints - scaled_sigmoid(xpoints, *params)) ** 2)

ax[0].set_title("FLOPs vs PC-1")
ax[0].set_xlabel("log10 FLOPs (1E21)")
ax[0].set_ylabel("PC-1")
ax[0].scatter(xpoints, ypoints)
ax[0].plot(xspace, y_sigmoid, color="red")
ax[0].text(0.1, 0.5, f"MSE: {sigmoid_mse:.2e}", transform=ax[0].transAxes)

xpoints = np.log10(base_llm_benchmark_eval["FLOPs_opt_Hoffman (1E21)"])
ypoints = base_llm_benchmark_eval["PC-1"]
# Fit a sigmoid to the data
params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0=(1, 0, 0, 0.555))
# plot
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
y_sigmoid = scaled_sigmoid(xspace, *params)

# compute MSE
sigmoid_mse = np.mean((ypoints - scaled_sigmoid(xpoints, *params)) ** 2)

ax[1].set_title("FLOPs_opt (Hoffman) vs PC-1")
ax[1].set_xlabel("log10 FLOPS_opt (1E21)")
ax[1].set_ylabel("PC-1")
ax[1].scatter(xpoints, ypoints)
ax[1].plot(xspace, y_sigmoid, color="red")
ax[1].text(0.1, 0.5, f"MSE: {sigmoid_mse:.2e}", transform=ax[1].transAxes)

xpoints = np.log10(base_llm_benchmark_eval["FLOPs_opt_Besiroglu (1E21)"])
ypoints = base_llm_benchmark_eval["PC-1"]
# Fit a sigmoid to the data
params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0=(1, 0, 0, 0.555))
# plot
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
y_sigmoid = scaled_sigmoid(xspace, *params)

# compute MSE
sigmoid_mse = np.mean((ypoints - scaled_sigmoid(xpoints, *params)) ** 2)

ax[2].set_title("FLOPs_opt (Besiroglu) vs PC-1")
ax[2].set_xlabel("log10 FLOPS_opt (1E21)")
ax[2].set_ylabel("PC-1")
ax[2].scatter(xpoints, ypoints)
ax[2].plot(xspace, y_sigmoid, color="red")
ax[2].text(0.1, 0.5, f"MSE: {sigmoid_mse:.2e}", transform=ax[2].transAxes)

plt.show()

# %%

############################################
# Plot a graph for each model family
############################################

families = duckdb.sql(
    """
    SELECT "Model Family" 
    FROM base_llm_benchmark_eval 
    GROUP BY "Model Family" 
    HAVING COUNT(*) >= 4
    """
).fetchall()
n_families = len(families)
ncols = 3
nrows = n_families // ncols + 1

fig, ax = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 6))  # 3 columns

axs = ax.flatten()
for i, family in enumerate(families):
    family = family[0]
    xpoints = np.log10(
        base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
            "FLOPs_opt_Besiroglu (1E21)"
        ]
    )
    ypoints = base_llm_benchmark_eval[
        base_llm_benchmark_eval["Model Family"] == family
    ]["PC-1"]
    # Fit a sigmoid to the data
    params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0=(1, 0, 0, 0.555))
    # Fit a line to the data
    lparams = np.polyfit(xpoints, ypoints, 1)
    # plot
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    y_sigmoid = scaled_sigmoid(xspace, *params)
    y_line = np.polyval(lparams, xspace)
    # compute MSE
    sigmoid_mse = np.mean((ypoints - scaled_sigmoid(xpoints, *params)) ** 2)
    linear_mse = np.mean((ypoints - np.polyval(lparams, xpoints)) ** 2)
    axs[i].set_title(f"{family} FLOPs_opt vs PC-1")
    axs[i].set_xlabel("log10 FLOPs_opt (1E21)")
    axs[i].set_ylabel("PC-1")
    axs[i].scatter(xpoints, ypoints)
    axs[i].plot(xspace, y_sigmoid, color="red", label="Sigmoid")
    axs[i].plot(xspace, y_line, color="green", label="Linear")
    axs[i].text(0.1, 0.9, f"Sigmoid MSE: {sigmoid_mse:.2e}", transform=axs[i].transAxes)
    axs[i].text(0.1, 0.8, f"Linear MSE: {linear_mse:.2e}", transform=axs[i].transAxes)
    axs[i].legend()

# %%

############################################
# Plot each model family on the same graph
############################################

families = duckdb.sql(
    """
    SELECT "Model Family" 
    FROM base_llm_benchmark_eval 
    GROUP BY "Model Family" 
    HAVING COUNT(*) >= 3
    """
).fetchall()

fig, ax = plt.subplots(1, 1, figsize=(14, 14))  # 1 columns

ax.set_title(f"FLOPs_opt vs PC-1")
ax.set_xlabel("log10 FLOPs_opt (1E21)")
ax.set_ylabel("PC-1")

for i, family in enumerate(families):
    family = family[0]
    xpoints = np.log10(
        base_llm_benchmark_eval[base_llm_benchmark_eval["Model Family"] == family][
            "FLOPs_opt_Besiroglu (1E21)"
        ]
    )
    ypoints = base_llm_benchmark_eval[
        base_llm_benchmark_eval["Model Family"] == family
    ]["PC-1"]
    # Fit a sigmoid to the data
    params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0=(1, 0, 0, 0.555))
    # Fit a line to the data
    lparams = np.polyfit(xpoints, ypoints, 1)
    # plot
    xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
    y_sigmoid = scaled_sigmoid(xspace, *params)
    y_line = np.polyval(lparams, xspace)
    # compute MSE
    sigmoid_mse = np.mean((ypoints - scaled_sigmoid(xpoints, *params)) ** 2)
    linear_mse = np.mean((ypoints - np.polyval(lparams, xpoints)) ** 2)

    ax.scatter(xpoints, ypoints, marker=utils.plot_markers.markers[i], label=family)
    ax.plot(xspace, y_sigmoid, color="red", label="Sigmoid")
    # ax.plot(xspace, y_line, color="green", label="Linear")
    # ax.text(0.1, 0.9, f"Sigmoid MSE: {sigmoid_mse:.2e}", transform=axs[i].transAxes)
    # ax.text(0.1, 0.8, f"Linear MSE: {linear_mse:.2e}", transform=axs[i].transAxes)

ax.legend()


family_release_dates = {
    "Yi-1.5": "2024-05-13",
    "Pythia": "2023-04-03",
    "SmolLM": "2024-07-16",
    "Qwen":   "2023-04-01",
    "BLOOM":  "2022-07-06",
    "CodeLlama": "2023-07-18",
    "Llama-2": "2023-07-18",
    "Qwen1.5": "2024-02-04",
    "Gemma-2": "2024-06-27",
    "OLMo":    "2024-02-01",
    "GPT-Neo/J": "2021-03-01",
    "StarCoder": "2023-05-04",
    "StarCoder2": "2023-02-28",
    "StableLM": "2023-08-05",
    "Cerebras-GPT": "2023-03-28",
    "H2O-Danube": "2024-02-28",
    "Llama": "2023-02-01",
    "DeepSeek-Coder": "2023-11-02",
    "OPT": "2022-05-03",
    "XGLM": "2021-12-20"
}

# write this to a file as CSV
df = pd.DataFrame(family_release_dates.items(), columns=["family", "release_date"])
df.to_csv("family_release_dates.csv", index=False)