# %%

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
        bounds=([0, -4, 0, 0], [10, 4, 1, 1]),
        maxfev=5000,
    )
    return popt


def plot_with_sigmoids(
    ax: plt.Axes,
    xlabel: str, 
    ylabel: str, 
    p0: tuple[
        # slope
        float, 
        # shift
        float, 
        # scale
        float, 
        # yoffset
        float
    ]
):
    ax.set_title(f"{xlabel} vs {ylabel}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xpoints_all = base_llm_benchmark_eval[xlabel]
    ypoints_all = base_llm_benchmark_eval[ylabel]

    xspace = np.linspace(min(xpoints_all), max(xpoints_all), 100)
    
    release_date_thresholds = [
        (2022.0, "green"),
        (2023.0, "yellow"),
        (2024.0, "orange"),
        (2025.0, "red"),
    ]

    for (thresh, color) in reversed(release_date_thresholds):    
        subset = base_llm_benchmark_eval[base_llm_benchmark_eval["release_date"] < thresh]
        ax.scatter(subset[xlabel], subset[ylabel], color=color)


    for i, (release_date_thresh, color) in enumerate(release_date_thresholds): 
        subset = base_llm_benchmark_eval[base_llm_benchmark_eval["release_date"] < release_date_thresh]
        xpoints = subset[xlabel]
        ypoints = subset[ylabel]
    
        scaled_sigmoid_params = get_scaled_sigmoid_parameters(xpoints, ypoints, p0)
        y_sigmoid = scaled_sigmoid(xspace, *scaled_sigmoid_params)

        sigmoid_mse = np.mean((ypoints_all - scaled_sigmoid(xpoints_all, *scaled_sigmoid_params)) ** 2)

        ax.plot(xspace, y_sigmoid, color=color, label=f"{release_date_thresh} sigmoid")
        yloc = 0.95 - 0.05*i
        ax.text(0.05, yloc, f"{release_date_thresh} MSE: {sigmoid_mse:.2e}", transform=ax.transAxes)
    
    ax.legend()


benchmarks = ["MMLU", "ARC-C", "HellaSwag", "Winograd", "TruthfulQA", "GSM8K", "XWinograd", "HumanEval"]

fig, ax = plt.subplots(len(benchmarks), 3, figsize=(21, len(benchmarks)*6))  # 3 columns

for i, benchmark in enumerate(benchmarks):
    plot_with_sigmoids(ax[i, 0], "log10 FLOPs (1E21)", benchmark, (1, 0, 0.5, 0.25))
    plot_with_sigmoids(ax[i, 1], "log10 FLOPs_opt_Besiroglu (1E21)", benchmark, (1, 0, 0.5, 0.25))
    plot_with_sigmoids(ax[i, 2], "PC-1", benchmark, (1, 0, 0.5, 0.25))
