# %%
from collections import defaultdict
from dataclasses import dataclass

import duckdb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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


# add flops and optimal flops to the dataframe
def augment_df_opt_flops(
    df: pd.DataFrame,
):
    # insert flops
    df["FLOP (1E21)"] = 6 * df["N"] * df["D"]
    # insert log flops
    df["log10 FLOP"] = np.log10(df["FLOP (1E21)"])

    param = EPOCH_PARAMS

    l_budgets = [
        loss(n * 1e9, d * 1e12, param)
        for n, d in zip(
            df["N"],
            df["D"],
        )
    ]
    n_opt, d_opt = zip(*[opt_params(l_budget, param) for l_budget in l_budgets])
    n_opt = np.array(n_opt)
    d_opt = np.array(d_opt)
    df["N_opt"] = n_opt / 1e9
    df["D_opt"] = d_opt / 1e12
    df["FLOP_opt"] = 6 * df["N_opt"] * df["D_opt"]
    df["log10 FLOP_opt"] = np.log10(df["FLOP_opt"])


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
        "Model Size (B)" as N, 
        "Pretraining Data Size (T)" as D, 
        hash("Model Family") as "family_idx"
    FROM base_llm_benchmark_eval
    JOIN family_release_dates ON base_llm_benchmark_eval."Model Family" = family_release_dates.family
    """
).df()
base_llm_benchmarks = ["MMLU", "ARC-C", "HellaSwag", "Winograd", "GSM8K", "XWinograd"]


augment_df_opt_flops(
    base_llm_benchmark_eval,
)

openllm_elo_merged = duckdb.read_csv("./data_models/meta/openllm_elo_merged.csv")
openllm_elo_merged = duckdb.sql(
    """
    SELECT
        "chatbot_arena_name",
        "arena_score" as Elo,
        "IFEval Raw",
        "BBH Raw",
        "MATH Lvl 5 Raw",
        "GPQA Raw",
        "MUSR Raw",
        "MMLU-PRO Raw",
        year(release_date) + (1/365)*dayofyear(release_date) as release_date,
        "N",
        "D",
    FROM openllm_elo_merged
    """
).df()
augment_df_opt_flops(openllm_elo_merged)
openllm_elo_benchmarks = [
    "IFEval Raw",
    "BBH Raw",
    "MATH Lvl 5 Raw",
    "GPQA Raw",
    "MUSR Raw",
    "MMLU-PRO Raw",
]


# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)
openllm_elo_merged.dropna(inplace=True)

benchmark_data = [
    ("MMLU", 0.25),
    ("ARC-C", 0.2),
    ("HellaSwag", 0.25),
    ("Winograd", 0.5),
    ("TruthfulQA", 0.5),
    ("GSM8K", 0.0),
    ("XWinograd", 0.5),
    ("HumanEval", 0.0),
    ("IFEval Raw", 0.0),
    ("BBH Raw", 0.25),
    ("MATH Lvl 5 Raw", 0.0),
    ("GPQA Raw", 0.25),
    ("MUSR Raw", 0.3),
    ("MMLU-PRO Raw", 0.1),
]
benchmark_floor_dict = defaultdict(lambda: 0.0, {b: f for b, f in benchmark_data})


def highest_score(df: pd.DataFrame, release_date, log10_flop_opt, key):
    """
    Returns the highest GPQA score achieved by a model with both a log10 FLOP_opt and release date under the specified value.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    release_date (datetime): The release date threshold.
    log10_flop_opt (float): The log10 FLOP_opt threshold.

    Returns:
    float: The highest GPQA score.
    """
    filtered_df = df[
        (df["release_date"] <= release_date) & (df["log10 FLOP_opt"] <= log10_flop_opt)
    ]
    if filtered_df.empty:
        return None
    return filtered_df[key].max()

def vectorized_highest_score(df, release_dates, log10_flop_opts, key):
    """
    Vectorized function to return the highest GPQA score for each combination of release_date and log10 FLOP_opt.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    release_dates (np.ndarray): Array of release date thresholds.
    log10_flop_opts (np.ndarray): Array of log10 FLOP_opt thresholds.
    key (str): The key to search for the highest score.

    Returns:
    np.ndarray: 2D array of highest GPQA scores.
    """
    # Create a 2D array to store the highest scores
    highest_scores = np.zeros((len(log10_flop_opts), len(release_dates)))

    for i, log10_flop_opt in enumerate(log10_flop_opts):
        mask = (df["log10 FLOP_opt"] <= log10_flop_opt)
        for j, release_date in enumerate(release_dates):
            combined_mask = mask & (df["release_date"] <= release_date)
            if combined_mask.any():
                highest_scores[i, j] = df.loc[combined_mask, key].max()
            else:
                highest_scores[i, j] = np.nan  # or some other placeholder for no data

    return highest_scores


# Perform a 2D linear fit
def linear_fit(x, a, b, c):
    return a * x[0] + b * x[1] + c


def plot_frontier(df: pd.DataFrame, key):
    plt.scatter(
        df["release_date"],
        df["log10 FLOP_opt"],
        c=df[key],
        cmap="viridis",
    )
    plt.colorbar(label=key)  # Add a colorbar to show the score scale

    plt.show()

    # Define a grid of release dates and log10 FLOP_opt values
    release_dates = np.linspace(
        df["release_date"].min(),
        df["release_date"].max(),
        100,
    )
    log10_flop_opts = np.linspace(
        df["log10 FLOP_opt"].min(),
        df["log10 FLOP_opt"].max(),
        100,
    )

    # Create a meshgrid
    X, Y = np.meshgrid(release_dates, log10_flop_opts)

    # Calculate the highest GPQA score for each point in the grid
    Z = vectorized_highest_score(df, release_dates, log10_flop_opts, key)
    
    # Plot the contourf plot
    plt.contourf(X, Y, Z, cmap="viridis")

    plt.scatter(
        df["release_date"],
        df["log10 FLOP_opt"],
        c=df[key],
        cmap="viridis",  # You can choose any colormap you prefer
    )

    plt.colorbar(label=key)
    plt.xlabel("Release Date")
    plt.ylabel("log10 FLOP_opt")
    plt.title("Frontier Plot")
    plt.show()
    

    
    # Fit the data
    xdata = np.stack((df["release_date"], df["log10 FLOP_opt"]))
    zdata = df[key]
    params, _ = curve_fit(linear_fit, xdata, zdata)
    a, b, c = params

    # Calculate the fitted values
    Z_fit = linear_fit(np.stack((X, Y)), a, b, c)
    
    plt.contourf(X, Y, Z_fit, cmap="viridis")
    
    plt.scatter(
        df["release_date"],
        df["log10 FLOP_opt"],
        c=df[key],
        cmap="viridis",  # You can choose any colormap you prefer
    )
    
    plt.colorbar(label=key)
    plt.xlabel("Release Date")
    plt.ylabel("log10 FLOP_opt")
    plt.title("Frontier Plot with Linear Fit")
    plt.show()


# %%

# plot_frontier(base_llm_benchmark_eval, "HumanEval")
plot_frontier(openllm_elo_merged, "Elo")