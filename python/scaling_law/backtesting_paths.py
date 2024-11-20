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

from util_obs_scaling_law_predictor import ScalingLaw
from util_timeseries_backtesting import ExpandingWindowBacktestSplitter
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
benchmark_floor_dict = {b: f for b, f in benchmark_data}
all_benchmarks = [b for b, _ in benchmark_data]


def add_logit_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, benchmarks: list[str], key: str
) -> LogitObsScalingLawPredictor:
    """
    Trains a logit model with the following benchmarks, and inserts a new column
    """
    benchmark_floor = [benchmark_floor_dict[b] for b in benchmarks]
    train_model_scores = torch.tensor(train_df[benchmarks].values, dtype=torch.float32)
    test_model_scores = torch.tensor(test_df[benchmarks].values, dtype=torch.float32)

    logit_obs_model = LogitObsScalingLawPredictor(
        benchmarks, benchmark_floor, train_model_scores
    )
    logit_obs_model.fit()

    train_df[key] = (
        logit_obs_model.predict_capability_scores(
            logit_obs_model.predict_logit_scores(train_model_scores)
        )
        .detach()
        .numpy()
    )
    test_df[key] = (
        logit_obs_model.predict_capability_scores(
            logit_obs_model.predict_logit_scores(test_model_scores)
        )
        .detach()
        .numpy()
    )

    return logit_obs_model


def add_linear_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, benchmarks: list[str], key: str
) -> LinearObsScalingLawPredictor:
    """
    Trains a linear model with the following benchmarks, and inserts a new column
    """
    train_model_scores = torch.tensor(train_df[benchmarks].values, dtype=torch.float32)
    test_model_scores = torch.tensor(test_df[benchmarks].values, dtype=torch.float32)

    linear_obs_model = LinearObsScalingLawPredictor(benchmarks, train_model_scores)
    linear_obs_model.fit()

    train_df[key] = (
        linear_obs_model.predict_capability_scores_from_model_scores(train_model_scores)
        .detach()
        .numpy()
    )

    test_df[key] = (
        linear_obs_model.predict_capability_scores_from_model_scores(test_model_scores)
        .detach()
        .numpy()
    )

    return linear_obs_model


#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs_splits = list(
    ExpandingWindowBacktestSplitter(
        min_train_size=40, test_size=20, increment=10, key="FLOPs_opt_Besiroglu (1E21)"
    ).split(base_llm_benchmark_eval)
)

ewbs_split_train_dict = {}
ewbs_split_test_dict = {}
ewbs_linear_model_dict = {}
ewbs_lin_slaw_dict = {}
ewbs_lin_slaw_err_dict = {}
ewbs_logit_model_dict = {}
ewbs_logit_slaw_dict = {}
ewbs_logit_slaw_err_dict = {}

for i, (train, test) in enumerate(ewbs_splits):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, test, benchmark_list, "PC-1 linear")
        logit_model = add_logit_model(train, test, benchmark_list, "PC-1 logit")

        # predict the excluded benchmark
        lin_slaw = ScalingLaw(floor=benchmark_floor_dict[excluded_benchmark])
        lin_slaw.fit(
            torch.tensor(train["PC-1 linear"].values, dtype=torch.float32),
            torch.tensor(train[excluded_benchmark].values, dtype=torch.float32),
        )

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # predict the excluded benchmark
        logit_slaw = ScalingLaw(floor=benchmark_floor_dict[excluded_benchmark])
        logit_slaw.fit(
            torch.tensor(train["PC-1 logit"].values, dtype=torch.float32),
            torch.tensor(train[excluded_benchmark].values, dtype=torch.float32),
        )

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # store the results
        ewbs_split_train_dict[(i, j)] = train
        ewbs_split_test_dict[(i, j)] = test
        ewbs_linear_model_dict[(i, j)] = linear_model
        ewbs_lin_slaw_dict[(i, j)] = lin_slaw
        ewbs_lin_slaw_err_dict[(i, j)] = lin_slaw_err
        ewbs_logit_model_dict[(i, j)] = logit_model
        ewbs_logit_slaw_dict[(i, j)] = logit_slaw
        ewbs_logit_slaw_err_dict[(i, j)] = logit_slaw_err

# %%

# create plot
fig, ax = plt.subplots(
    len(ewbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(ewbs_splits)),
)

for i in range(len(ewbs_splits)):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        train = ewbs_split_train_dict[(i, j)]
        test = ewbs_split_test_dict[(i, j)]
        lin_slaw = ewbs_lin_slaw_dict[(i, j)]
        lin_slaw_err = ewbs_lin_slaw_err_dict[(i, j)]
        logit_slaw = ewbs_logit_slaw_dict[(i, j)]
        logit_slaw_err = ewbs_logit_slaw_err_dict[(i, j)]
        # Plot Train ( x marker)

        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            train[excluded_benchmark],
            label="True",
            color="black",
            marker="x",
            alpha=0.5,
        )
        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(train["PC-1 linear"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear",
            color="blue",
            marker="x",
            alpha=0.5,
        )
        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(train["PC-1 logit"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit",
            color="red",
            marker="x",
            alpha=0.5,
        )

        # Plot Test

        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            test[excluded_benchmark],
            label="True",
            color="black",
            alpha=0.5,
        )
        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear, MSE: {lin_slaw_err:.3f}",
            color="blue",
            alpha=0.5,
        )
        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit, MSE: {logit_slaw_err:.3f}",
            color="red",
            alpha=0.5,
        )
        ax[i, j].set_title(f"{excluded_benchmark} (train size: {len(train)})")
        ax[i, j].legend()


# print the mean error
err_lin = np.zeros((len(ewbs_splits), len(all_benchmarks)))
err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))

for i in range(len(ewbs_splits)):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        err_lin[i, j] = ewbs_lin_slaw_err_dict[(i, j)]
        err_logit[i, j] = ewbs_logit_slaw_err_dict[(i, j)]

print(f"Expanding Window Mean Linear Error: {err_lin.mean()}")
print(f"Expanding Window Mean Logit Error: {err_logit.mean()}")

print(f"Expanding Window Percent improvement: {100*(err_lin.mean() - err_logit.mean())/err_lin.mean()}")

#####################################
# Train and fit global linear and logit models of PC-1
# Rolling Window
#####################################

rwbs_splits = list(
    ExpandingWindowBacktestSplitter(
        min_train_size=40, test_size=20, increment=10, key="FLOPs_opt_Besiroglu (1E21)"
    ).split(base_llm_benchmark_eval)
)

rwbs_split_train_dict = {}
rwbs_split_test_dict = {}
rwbs_linear_model_dict = {}
rwbs_lin_slaw_dict = {}
rwbs_lin_slaw_err_dict = {}
rwbs_logit_model_dict = {}
rwbs_logit_slaw_dict = {}
rwbs_logit_slaw_err_dict = {}

for i, (train, test) in enumerate(rwbs_splits):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, test, benchmark_list, "PC-1 linear")
        logit_model = add_logit_model(train, test, benchmark_list, "PC-1 logit")

        # predict the excluded benchmark
        lin_slaw = ScalingLaw(floor=benchmark_floor_dict[excluded_benchmark])
        lin_slaw.fit(
            torch.tensor(train["PC-1 linear"].values, dtype=torch.float32),
            torch.tensor(train[excluded_benchmark].values, dtype=torch.float32),
        )

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # predict the excluded benchmark
        logit_slaw = ScalingLaw(floor=benchmark_floor_dict[excluded_benchmark])
        logit_slaw.fit(
            torch.tensor(train["PC-1 logit"].values, dtype=torch.float32),
            torch.tensor(train[excluded_benchmark].values, dtype=torch.float32),
        )

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # store the results
        rwbs_split_train_dict[(i, j)] = train
        rwbs_split_test_dict[(i, j)] = test
        rwbs_linear_model_dict[(i, j)] = linear_model
        rwbs_lin_slaw_dict[(i, j)] = lin_slaw
        rwbs_lin_slaw_err_dict[(i, j)] = lin_slaw_err
        rwbs_logit_model_dict[(i, j)] = logit_model
        rwbs_logit_slaw_dict[(i, j)] = logit_slaw
        rwbs_logit_slaw_err_dict[(i, j)] = logit_slaw_err

# %%

# create plot
fig, ax = plt.subplots(
    len(rwbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(rwbs_splits)),
)


for i in range(len(rwbs_splits)):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        train = rwbs_split_train_dict[(i, j)]
        test = rwbs_split_test_dict[(i, j)]
        lin_slaw = rwbs_lin_slaw_dict[(i, j)]
        lin_slaw_err = rwbs_lin_slaw_err_dict[(i, j)]
        logit_slaw = rwbs_logit_slaw_dict[(i, j)]
        logit_slaw_err = rwbs_logit_slaw_err_dict[(i, j)]
        # Plot Train ( x marker)

        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            train[excluded_benchmark],
            label="True",
            color="black",
            marker="x",
            alpha=0.5,
        )
        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(train["PC-1 linear"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear",
            color="blue",
            marker="x",
            alpha=0.5,
        )
        ax[i, j].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(train["PC-1 logit"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit",
            color="red",
            marker="x",
            alpha=0.5,
        )

        # Plot Test

        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            test[excluded_benchmark],
            label="True",
            color="black",
            alpha=0.5,
        )
        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear, MSE: {lin_slaw_err:.3f}",
            color="blue",
            alpha=0.5,
        )
        ax[i, j].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit, MSE: {logit_slaw_err:.3f}",
            color="red",
            alpha=0.5,
        )
        ax[i, j].set_title(f"{excluded_benchmark} (train size: {len(train)})")
        ax[i, j].legend()


# print the mean error
err_lin = np.zeros((len(ewbs_splits), len(all_benchmarks)))
err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))

for i in range(len(ewbs_splits)):
    for j, excluded_benchmark in enumerate(all_benchmarks):
        err_lin[i, j] = ewbs_lin_slaw_err_dict[(i, j)]
        err_logit[i, j] = ewbs_logit_slaw_err_dict[(i, j)]

print(f"Rolling Window Mean Linear Error: {err_lin.mean()}")
print(f"Rolling Window Mean Logit Error: {err_logit.mean()}")

print(f"Rolling Window Percent improvement: {100*(err_lin.mean() - err_logit.mean())/err_lin.mean()}")

#####################################
# Train and fit family-specific linear models of PC-1
#####################################
