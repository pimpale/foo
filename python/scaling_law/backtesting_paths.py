# %%
import time
import duckdb
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.axes
from tqdm import tqdm

from util_obs_scaling_law_predictor import ObsScalingLawPredictor, ScalingLaw
from util_timeseries_backtesting import (
    ExpandingWindowBacktestSplitter,
    RollingWindowBacktestSplitter,
)
from util_linear_obs_scaling_law_predictor import LinearObsScalingLawPredictor
from util_logit_obs_scaling_law_predictor import LogitObsScalingLawPredictor

torch.set_num_threads(1)


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
    train_df: pd.DataFrame, benchmarks: list[str]
) -> LogitObsScalingLawPredictor:
    """
    Trains a logit model with the following benchmarks, and inserts a new column
    """
    benchmark_floor = [benchmark_floor_dict[b] for b in benchmarks]
    train_model_scores = torch.tensor(train_df[benchmarks].values, dtype=torch.float32)

    logit_obs_model = LogitObsScalingLawPredictor(
        benchmarks, benchmark_floor, train_model_scores
    )
    t0 = time.time()
    logit_obs_model.fit()
    logit_obs_model.eval()
    print(f"Logit Training Time: {time.time() - t0:.2f} seconds")
    return logit_obs_model


def add_linear_model(
    train_df: pd.DataFrame, benchmarks: list[str]
) -> LinearObsScalingLawPredictor:
    """
    Trains a linear model with the following benchmarks, and inserts a new column
    """
    train_model_scores = torch.tensor(train_df[benchmarks].values, dtype=torch.float32)

    linear_obs_model = LinearObsScalingLawPredictor(benchmarks, train_model_scores)
    t0 = time.time()
    linear_obs_model.fit()
    linear_obs_model.eval()
    print(f"Linear Training Time: {time.time() - t0:.2f} seconds")

    return linear_obs_model


def add_slaw(
    train: pd.DataFrame,
    model: ObsScalingLawPredictor,
    benchmark_key: str,
) -> ScalingLaw:
    model_scores = torch.tensor(train[model.benchmarks].values, dtype=torch.float32)
    capability_scores = model.predict_capability_scores_from_model_scores(
        model_scores
    ).detach()
    benchmark_scores = torch.tensor(train[benchmark_key].values, dtype=torch.float32)
    slaw = ScalingLaw(
        benchmark=benchmark_key,
        floor=benchmark_floor_dict[benchmark_key],
        capability_scores=capability_scores,
        benchmark_scores=benchmark_scores,
    )
    t0 = time.time()
    slaw.fit()
    slaw.eval()
    print(f"Scaling Law Training Time: {time.time() - t0:.2f} seconds")
    return slaw


@dataclass
class Spe:
    """
    Scatter Plot Entry
    """

    y_key: str
    color: str


def plot_train_test(
    ax: matplotlib.axes.Axes,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    x_key: str,
    entries: list[Spe],
    title=None,
    y_label=None,
):
    for e in entries:
        ax.scatter(
            train_df[x_key],
            train_df[e.y_key],
            label="Train",
            marker="x",
            alpha=0.5,
            color=e.color,
        )
        ax.scatter(
            test_df[x_key],
            test_df[e.y_key],
            label="Test",
            marker="o",
            alpha=0.5,
            color=e.color,
        )
    ax.legend()

    ax.set_xlabel(x_key)
    if y_label is not None:
        ax.set_ylabel(y_label)
    if title is None and (x_key is not None and y_label is not None):
        title = f"{y_label} vs {x_key}"
    if title is not None:
        ax.set_title(title)


def plot_linear_model(
    ax_arr: np.ndarray,
    bench_idx: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    linear_obs_model: LinearObsScalingLawPredictor,
):
    benchmark = linear_obs_model.benchmarks[bench_idx]
    plot_train_test(
        ax_arr[0],
        train,
        test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred", "C1"),
        ],
        y_label=benchmark,
    )
    plot_train_test(
        ax_arr[1],
        train,
        test,
        "PC-1 (linear)",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred", "C1"),
        ],
    )


def plot_logit_model(
    ax_arr: np.ndarray,
    bench_idx: int,
    train: pd.DataFrame,
    test: pd.DataFrame,
    logit_obs_model: LogitObsScalingLawPredictor,
):
    benchmark = logit_obs_model.benchmarks[bench_idx]
    plot_train_test(
        ax_arr[0],
        train,
        test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred", "C1"),
        ],
        y_label=benchmark,
    )

    plot_train_test(
        ax_arr[1],
        train,
        test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{benchmark} logit", "C0"),
            Spe(f"{benchmark} logit pred", "C1"),
        ],
        y_label=f"{benchmark} logit",
    )

    plot_train_test(
        ax_arr[2],
        train,
        test,
        "PC-1 (logit)",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred", "C1"),
        ],
        y_label=benchmark,
    )

    plot_train_test(
        ax_arr[3],
        train,
        test,
        "PC-1 (logit)",
        [
            Spe(f"{benchmark} logit", "C0"),
            Spe(f"{benchmark} logit pred", "C1"),
        ],
        y_label=f"{benchmark} logit",
    )


def augment_df_logit(
    logit_obs_model: LogitObsScalingLawPredictor, df_to_augment: pd.DataFrame
):
    x = torch.tensor(
        df_to_augment[logit_obs_model.benchmarks].values, dtype=torch.float32
    )
    x_logit = logit_obs_model.predict_logit_scores(x)
    capability_score = logit_obs_model.predict_capability_scores(x_logit)
    x_hat_logit = logit_obs_model.predict_benchmark_logit_scores(capability_score)
    x_hat = logit_obs_model.predict_benchmark_scores(x_hat_logit)

    df_to_augment["PC-1 (logit)"] = capability_score.detach().numpy()

    for b_idx, benchmark in enumerate(logit_obs_model.benchmarks):
        df_to_augment[f"{benchmark} logit"] = x_logit.T[b_idx].detach().numpy()
        df_to_augment[f"{benchmark} logit pred"] = x_hat_logit.T[b_idx].detach().numpy()
        df_to_augment[f"{benchmark} pred"] = x_hat.T[b_idx].detach().numpy()


def augment_test_train_logit(
    logit_obs_model: LogitObsScalingLawPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_logit(logit_obs_model, train)
    augment_df_logit(logit_obs_model, test)


def augment_df_linear(
    linear_obs_model: LinearObsScalingLawPredictor, df_to_augment: pd.DataFrame
):
    x = torch.tensor(
        df_to_augment[linear_obs_model.benchmarks].values, dtype=torch.float32
    )
    capability_score = linear_obs_model.predict_capability_scores_from_model_scores(x)
    x_hat = linear_obs_model.predict_benchmark_scores_from_capability_scores(
        capability_score
    )

    df_to_augment["PC-1 (linear)"] = capability_score.detach().numpy()

    for b_idx, benchmark in enumerate(linear_obs_model.benchmarks):
        df_to_augment[f"{benchmark} pred"] = x_hat.T[b_idx].detach().numpy()


def augment_test_train_linear(
    linear_obs_model: LinearObsScalingLawPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_linear(linear_obs_model, train)
    augment_df_linear(linear_obs_model, test)


def augment_df_slaw(
    slaw: ScalingLaw, model: ObsScalingLawPredictor, df_to_augment: pd.DataFrame
):
    model_scores = torch.tensor(
        df_to_augment[model.benchmarks].values, dtype=torch.float32
    )
    benchmark_scores = torch.tensor(
        df_to_augment[slaw.benchmark].values, dtype=torch.float32
    )
    capability_scores = model.predict_capability_scores_from_model_scores(
        model_scores
    ).detach()

    df_to_augment["PC-1"] = capability_scores.numpy()

    df_to_augment[f"{slaw.benchmark} logit"] = (
        slaw.predict_logit_scores(benchmark_scores).detach().numpy()
    )
    df_to_augment[f"{slaw.benchmark} logit pred"] = (
        slaw.predict_benchmark_logit_scores(capability_scores).detach().numpy()
    )
    df_to_augment[f"{slaw.benchmark} pred"] = (
        slaw.forward(capability_scores).detach().numpy()
    )


def augment_test_train_slaw(
    slaw: ScalingLaw,
    model: ObsScalingLawPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_slaw(slaw, model, train)
    augment_df_slaw(slaw, model, test)


# %%

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
ewbs_lin_slaw_dict: dict[tuple[int, int], ScalingLaw] = {}
ewbs_logit_model_dict = {}
ewbs_logit_slaw_dict: dict[tuple[int, int], ScalingLaw] = {}

n_trains = len(ewbs_splits) * len(all_benchmarks)

for split_idx, (train, test) in enumerate(ewbs_splits):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        i_train = split_idx * len(all_benchmarks) + bench_idx
        print(f"Training {i_train}/{n_trains}")
        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, benchmark_list)
        logit_model = add_logit_model(train, benchmark_list)

        # predict the excluded benchmark
        lin_slaw = add_slaw(train, linear_model, excluded_benchmark)
        logit_slaw = add_slaw(train, logit_model, excluded_benchmark)

        # store the results
        ewbs_split_train_dict[(split_idx, bench_idx)] = train
        ewbs_split_test_dict[(split_idx, bench_idx)] = test
        ewbs_linear_model_dict[(split_idx, bench_idx)] = linear_model
        ewbs_lin_slaw_dict[(split_idx, bench_idx)] = lin_slaw
        ewbs_logit_model_dict[(split_idx, bench_idx)] = logit_model
        ewbs_logit_slaw_dict[(split_idx, bench_idx)] = logit_slaw

# %%

# create plot
fig, ax = plt.subplots(
    len(ewbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(ewbs_splits)),
    squeeze=False,
)


# print the mean error
e_err_lin = np.zeros((len(ewbs_splits), len(all_benchmarks)))
e_err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))

for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        train = ewbs_split_train_dict[(split_idx, bench_idx)].copy()
        test = ewbs_split_test_dict[(split_idx, bench_idx)].copy()
        linear_model = ewbs_linear_model_dict[(split_idx, bench_idx)]
        logit_model = ewbs_logit_model_dict[(split_idx, bench_idx)]
        lin_slaw = ewbs_lin_slaw_dict[(split_idx, bench_idx)]
        logit_slaw = ewbs_logit_slaw_dict[(split_idx, bench_idx)]

        # augment the df with columns
        augment_test_train_linear(linear_model, train, test)
        augment_test_train_logit(logit_model, train, test)

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 (linear)"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 (logit)"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        e_err_lin[split_idx, bench_idx] = lin_slaw_err
        e_err_logit[split_idx, bench_idx] = logit_slaw_err

        # Plot Train ( x marker)
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            train[excluded_benchmark],
            label="True",
            color="black",
            marker="x",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(train["PC-1 (linear)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear",
            color="blue",
            marker="x",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(train["PC-1 (logit)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit",
            color="red",
            marker="x",
            alpha=0.5,
        )

        # Plot Test

        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            test[excluded_benchmark],
            label="True",
            color="black",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(test["PC-1 (linear)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear, MSE: {lin_slaw_err:.3f}",
            color="blue",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(test["PC-1 (logit)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit, MSE: {logit_slaw_err:.3f}",
            color="red",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].set_title(
            f"{excluded_benchmark} (train size: {len(train)})"
        )
        ax[split_idx, bench_idx].legend()

print(f"Expanding Window Mean Linear Error: {e_err_lin.mean()}")
print(f"Expanding Window Mean Logit Error: {e_err_logit.mean()}")

print(
    f"Expanding Window Percent improvement: {100*(e_err_lin.mean() - e_err_logit.mean())/e_err_lin.mean()}"
)

# %%
split_idx = 2
bench_idx = 0
linear_obs_model = ewbs_linear_model_dict[(split_idx, bench_idx)]
train = ewbs_split_train_dict[(split_idx, bench_idx)]
test = ewbs_split_test_dict[(split_idx, bench_idx)]
excluded_benchmark = all_benchmarks[bench_idx]
lin_slaw = ewbs_lin_slaw_dict[(split_idx, bench_idx)]

fig, ax = plt.subplots(
    len(linear_obs_model.benchmarks),
    2,
    figsize=(10, len(linear_obs_model.benchmarks) * 5),
    squeeze=False,
)  # 1 columns

# insert data from excluded benchmark

for bench_idx, benchmark in enumerate(linear_obs_model.benchmarks):
    train_copy, test_copy = train.copy(), test.copy()
    augment_test_train_linear(linear_obs_model, train_copy, test_copy)
    plot_linear_model(ax[bench_idx], bench_idx, train_copy, test_copy, linear_obs_model)

# now plot the data for the actual fit curve on the excluded benchmark
# 1 row, 4 columns
# col 0: FLOPs vs benchmark (show both true and predicted)
# col 1: FLOPs vs logit benchmark (show both true and predicted)
# col 2: capability vs benchmark (show both true and predicted)
# col 3: capability vs logit benchmark (show both true and predicted)


train_copy, test_copy = train.copy(), test.copy()
augment_test_train_slaw(lin_slaw, linear_obs_model, train_copy, test_copy)

fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
ax_arr = ax[0]
# plot in flop x-space and benchmark y-space
plot_train_test(
    ax_arr[0],
    train_copy,
    test_copy,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[1],
    train_copy,
    test_copy,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)

plot_train_test(
    ax_arr[2],
    train_copy,
    test_copy,
    "PC-1",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[3],
    train_copy,
    test_copy,
    "PC-1",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)

plt.show()
# %%
split_idx = 0
bench_idx = 1
logit_obs_model = ewbs_logit_model_dict[(split_idx, bench_idx)]
train = ewbs_split_train_dict[(split_idx, bench_idx)]
test = ewbs_split_test_dict[(split_idx, bench_idx)]
excluded_benchmark = all_benchmarks[bench_idx]
logit_slaw = ewbs_logit_slaw_dict[(split_idx, bench_idx)]

fig, ax = plt.subplots(
    len(logit_obs_model.benchmarks),
    4,
    figsize=(4 * 4, len(logit_obs_model.benchmarks) * 4),
    squeeze=False,
)  # 1 columns

for bench_idx, benchmark in enumerate(logit_obs_model.benchmarks):
    train_copy, test_copy = train.copy(), test.copy()
    augment_test_train_logit(logit_obs_model, train_copy, test_copy)
    plot_logit_model(ax[bench_idx], bench_idx, train_copy, test_copy, logit_obs_model)

plt.tight_layout()

plt.show()

train_copy, test_copy = train.copy(), test.copy()
augment_test_train_slaw(logit_slaw, logit_obs_model, train_copy, test_copy)

fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
ax_arr = ax[0]
# plot in flop x-space and benchmark y-space
plot_train_test(
    ax_arr[0],
    train_copy,
    test_copy,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[1],
    train_copy,
    test_copy,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)

plot_train_test(
    ax_arr[2],
    train_copy,
    test_copy,
    "PC-1",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[3],
    train_copy,
    test_copy,
    "PC-1",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)

plt.show()

# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Rolling Window
#####################################

rwbs_splits = list(
    RollingWindowBacktestSplitter(
        train_size=40, test_size=20, increment=10, key="FLOPs_opt_Besiroglu (1E21)"
    ).split(base_llm_benchmark_eval)
)

rwbs_split_train_dict = {}
rwbs_split_test_dict = {}
rwbs_linear_model_dict = {}
rwbs_lin_slaw_dict = {}
rwbs_logit_model_dict = {}
rwbs_logit_slaw_dict = {}

n_trains = len(rwbs_splits) * len(all_benchmarks)

for split_idx, (train, test) in enumerate(rwbs_splits):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        i_train = split_idx * len(all_benchmarks) + bench_idx
        print(f"Training {i_train}/{n_trains}")

        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, benchmark_list)
        logit_model = add_logit_model(train, benchmark_list)

        # predict the excluded benchmark
        lin_slaw = add_slaw(train, linear_model, excluded_benchmark)
        logit_slaw = add_slaw(train, logit_model, excluded_benchmark)

        # store the results
        rwbs_split_train_dict[(split_idx, bench_idx)] = train
        rwbs_split_test_dict[(split_idx, bench_idx)] = test
        rwbs_linear_model_dict[(split_idx, bench_idx)] = linear_model
        rwbs_lin_slaw_dict[(split_idx, bench_idx)] = lin_slaw
        rwbs_logit_model_dict[(split_idx, bench_idx)] = logit_model
        rwbs_logit_slaw_dict[(split_idx, bench_idx)] = logit_slaw

# %%

# create plot
fig, ax = plt.subplots(
    len(rwbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(rwbs_splits)),
)


# print the mean error
r_err_lin = np.zeros((len(rwbs_splits), len(all_benchmarks)))
r_err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))


for split_idx in range(len(rwbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        train = rwbs_split_train_dict[(split_idx, bench_idx)]
        test = rwbs_split_test_dict[(split_idx, bench_idx)]
        lin_slaw = rwbs_lin_slaw_dict[(split_idx, bench_idx)]
        logit_slaw = rwbs_logit_slaw_dict[(split_idx, bench_idx)]
        linear_model = rwbs_linear_model_dict[(split_idx, bench_idx)]
        logit_model = rwbs_logit_model_dict[(split_idx, bench_idx)]

        # augment the df with columns
        augment_test_train_linear(linear_model, train, test)
        augment_test_train_logit(logit_model, train, test)

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 (linear)"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 (logit)"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        r_err_lin[split_idx, bench_idx] = lin_slaw_err
        r_err_logit[split_idx, bench_idx] = logit_slaw_err

        # Plot Train ( x marker)

        # Plot Train ( x marker)
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            train[excluded_benchmark],
            label="True",
            color="black",
            marker="x",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(train["PC-1 (linear)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear",
            color="blue",
            marker="x",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            train["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(train["PC-1 (logit)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit",
            color="red",
            marker="x",
            alpha=0.5,
        )

        # Plot Test

        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            test[excluded_benchmark],
            label="True",
            color="black",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            lin_slaw.forward(
                torch.tensor(test["PC-1 (linear)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Linear, MSE: {lin_slaw_err:.3f}",
            color="blue",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].scatter(
            test["log10 FLOPs_opt_Besiroglu (1E21)"],
            logit_slaw.forward(
                torch.tensor(test["PC-1 (logit)"].values, dtype=torch.float32)
            )
            .detach()
            .numpy(),
            label=f"Logit, MSE: {logit_slaw_err:.3f}",
            color="red",
            alpha=0.5,
        )
        ax[split_idx, bench_idx].set_title(
            f"{excluded_benchmark} (train size: {len(train)})"
        )
        ax[split_idx, bench_idx].legend()


print(f"Rolling Window Mean Linear Error: {r_err_lin.mean()}")
print(f"Rolling Window Mean Logit Error: {r_err_logit.mean()}")

print(
    f"Rolling Window Percent improvement: {100*(r_err_lin.mean() - r_err_logit.mean())/r_err_lin.mean()}"
)

# %%

# plot the distribution of betas
fig, ax = plt.subplots(5, 1, figsize=(5, 14))
logit_betas = []
linear_betas = []
logit_alphas = []
linear_alphas = []
logit_ceil_raws = []
for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        logit_model = ewbs_logit_model_dict[(split_idx, bench_idx)]
        linear_model = ewbs_linear_model_dict[(split_idx, bench_idx)]
        logit_betas.extend(logit_model.beta.detach().numpy())
        linear_betas.extend(linear_model.benchmark_weights.detach().numpy())
        logit_alphas.extend(logit_model.alpha.detach().numpy())
        linear_alphas.extend(linear_model.alpha.detach().numpy())
        logit_ceil_raws.extend(logit_model.benchmark_ceil_raw.detach().numpy())

ax[0].hist(logit_betas, bins=20, alpha=0.5, label="Logit Betas")
ax[0].set_title("Logit Betas")
ax[0].legend()

ax[1].hist(linear_betas, bins=20, alpha=0.5, label="Linear Betas")
ax[1].set_title("Linear Betas")
ax[1].legend()

ax[2].hist(logit_alphas, bins=20, alpha=0.5, label="Logit Alphas")
ax[2].set_title("Logit Alphas")
ax[2].legend()

ax[3].hist(linear_alphas, bins=20, alpha=0.5, label="Linear Alphas")
ax[3].set_title("Linear Alphas")
ax[3].legend()

ax[4].hist(logit_ceil_raws, bins=20, alpha=0.5, label="Logit Ceil Raw")
ax[4].set_title("Logit Ceil Raw")
ax[4].legend()

# %%

# plot the distribution of scaling law parameters
fig, ax = plt.subplots(6, 1, figsize=(5, 14))

lin_betas = []
lin_alphas = []
lin_ceil_raws = []

logit_betas = []
logit_alphas = []
logit_ceil_raws = []
for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        lin_slaw = ewbs_lin_slaw_dict[(split_idx, bench_idx)]
        logit_slaw = ewbs_logit_slaw_dict[(split_idx, bench_idx)]

        lin_betas.append(lin_slaw.beta.detach().numpy())
        lin_alphas.append(lin_slaw.alpha.detach().numpy())
        lin_ceil_raws.append(lin_slaw.benchmark_ceil_raw.detach().numpy())

        logit_betas.append(logit_slaw.beta.detach().numpy())
        logit_alphas.append(logit_slaw.alpha.detach().numpy())
        logit_ceil_raws.append(logit_slaw.benchmark_ceil_raw.detach().numpy())

ax[0].hist(lin_betas, bins=20, alpha=0.5, label="Linear Betas")
ax[0].set_title("Linear Betas")
ax[0].legend()

ax[1].hist(logit_betas, bins=20, alpha=0.5, label="Logit Betas")
ax[1].set_title("Logit Betas")
ax[1].legend()

ax[2].hist(lin_alphas, bins=20, alpha=0.5, label="Linear Alphas")
ax[2].set_title("Linear Alphas")
ax[2].legend()

ax[3].hist(logit_alphas, bins=20, alpha=0.5, label="Logit Alphas")
ax[3].set_title("Logit Alphas")
ax[3].legend()

ax[4].hist(lin_ceil_raws, bins=20, alpha=0.5, label="Linear Ceil Raw")
ax[4].set_title("Linear Ceil Raw")
ax[4].legend()

ax[5].hist(logit_ceil_raws, bins=20, alpha=0.5, label="Logit Ceil Raw")
ax[5].set_title("Logit Ceil Raw")
ax[5].legend()


# %%

fig, ax = plt.subplots(
    len(ewbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(ewbs_splits)),
    squeeze=False,
)

# Plot all loss curves for logit training
for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        logit_model = ewbs_logit_model_dict[(split_idx, bench_idx)]
        ax[split_idx, bench_idx].plot(
            np.log(logit_model.train_losses[100:]),
            label=f"Split: {split_idx}, Bench: {bench_idx}",
        )

        linear_model = ewbs_linear_model_dict[(split_idx, bench_idx)]
        ax[split_idx, bench_idx].plot(
            np.log(linear_model.train_losses[100:]),
            label=f"Split: {split_idx}, Bench: {bench_idx}",
        )
# %%

fig, ax = plt.subplots(
    len(ewbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(ewbs_splits)),
    squeeze=False,
)

# Plot all loss curves for logit training
for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        logit_slaw = ewbs_logit_slaw_dict[(split_idx, bench_idx)]
        ax[split_idx, bench_idx].plot(
            np.log(logit_slaw.train_losses[0:]),
            label=f"Split: {split_idx}, Bench: {bench_idx}",
        )

        lin_slaw = ewbs_lin_slaw_dict[(split_idx, bench_idx)]
        ax[split_idx, bench_idx].plot(
            np.log(lin_slaw.train_losses[0:]),
            label=f"Split: {split_idx}, Bench: {bench_idx}",
        )


# %%
#####################################
# Train and fit family-specific linear models of PC-1
#####################################
