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

from util_obs_scaling_law_predictor import ScalingLaw
from util_timeseries_backtesting import (
    ExpandingWindowBacktestSplitter,
    RollingWindowBacktestSplitter,
)
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


def add_slaw(
    train: pd.DataFrame,
    test: pd.DataFrame,
    benchmark_key: str,
    capability_key: str,
) -> ScalingLaw:
    train_capability_scores = torch.tensor(
        train[capability_key].values, dtype=torch.float32
    )
    test_capability_scores = torch.tensor(
        test[capability_key].values, dtype=torch.float32
    )
    slaw = ScalingLaw(
        floor=benchmark_floor_dict[benchmark_key],
        capability_scores=train_capability_scores,
        benchmark_scores=torch.tensor(train[benchmark_key].values, dtype=torch.float32),
    )
    t0 = time.time()
    slaw.fit()
    print(f"{capability_key} Law Training Time: {time.time() - t0:.2f} seconds")

    train[f"{benchmark_key} logit pred ({capability_key})"] = (
        slaw.predict_benchmark_logit_scores(train_capability_scores).detach().numpy()
    )
    train[f"{benchmark_key} pred ({capability_key})"] = (
        slaw.forward(train_capability_scores).detach().numpy()
    )
    test[f"{benchmark_key} logit pred ({capability_key})"] = (
        slaw.predict_benchmark_logit_scores(test_capability_scores).detach().numpy()
    )
    test[f"{benchmark_key} pred ({capability_key})"] = (
        slaw.forward(test_capability_scores).detach().numpy()
    )

    return slaw


def add_logit_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, benchmarks: list[str]
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
    t0 = time.time()
    logit_obs_model.fit()
    print(f"Logit Training Time: {time.time() - t0:.2f} seconds")

    train_logit_scores = logit_obs_model.predict_logit_scores(train_model_scores)
    train_capability_scores = logit_obs_model.predict_capability_scores(
        train_logit_scores
    )
    train_benchmark_logit_scores = logit_obs_model.predict_benchmark_logit_scores(
        train_capability_scores
    )
    train_benchmark_scores = logit_obs_model.predict_benchmark_scores(
        train_benchmark_logit_scores
    )

    for b_idx, benchmark in enumerate(benchmarks):
        train_df[f"{benchmark} logit"] = train_logit_scores.T[b_idx].detach().numpy()
        train_df[f"{benchmark} logit pred"] = (
            train_benchmark_logit_scores.T[b_idx].detach().numpy()
        )
        train_df[f"{benchmark} pred (logit)"] = (
            train_benchmark_scores.T[b_idx].detach().numpy()
        )

    train_df["PC-1 logit"] = train_capability_scores.detach().numpy()

    test_logit_scores = logit_obs_model.predict_logit_scores(test_model_scores)
    test_capability_scores = logit_obs_model.predict_capability_scores(
        test_logit_scores
    )
    test_benchmark_logit_scores = logit_obs_model.predict_benchmark_logit_scores(
        test_capability_scores
    )
    test_benchmark_scores = logit_obs_model.predict_benchmark_scores(
        test_benchmark_logit_scores
    )

    for b_idx, benchmark in enumerate(benchmarks):
        test_df[f"{benchmark} logit"] = test_logit_scores.T[b_idx].detach().numpy()
        test_df[f"{benchmark} logit pred"] = (
            test_benchmark_logit_scores.T[b_idx].detach().numpy()
        )
        test_df[f"{benchmark} pred (logit)"] = (
            test_benchmark_scores.T[b_idx].detach().numpy()
        )

    test_df["PC-1 logit"] = test_capability_scores.detach().numpy()

    return logit_obs_model


def add_linear_model(
    train_df: pd.DataFrame, test_df: pd.DataFrame, benchmarks: list[str]
) -> LinearObsScalingLawPredictor:
    """
    Trains a linear model with the following benchmarks, and inserts a new column
    """
    train_model_scores = torch.tensor(train_df[benchmarks].values, dtype=torch.float32)
    test_model_scores = torch.tensor(test_df[benchmarks].values, dtype=torch.float32)

    linear_obs_model = LinearObsScalingLawPredictor(benchmarks, train_model_scores)
    t0 = time.time()
    linear_obs_model.fit()
    print(f"Linear Training Time: {time.time() - t0:.2f} seconds")

    train_model_scores = linear_obs_model.train_model_scores
    train_capability_scores = (
        linear_obs_model.predict_capability_scores_from_model_scores(train_model_scores)
    )
    train_benchmark_scores = (
        linear_obs_model.predict_benchmark_scores_from_capability_scores(
            train_capability_scores
        )
    )

    for b_idx, benchmark in enumerate(benchmarks):
        train_df[f"{benchmark} pred (linear)"] = (
            train_benchmark_scores.T[b_idx].detach().numpy()
        )

    train_df["PC-1 linear"] = train_capability_scores.detach().numpy()

    test_capability_scores = (
        linear_obs_model.predict_capability_scores_from_model_scores(test_model_scores)
    )
    test_benchmark_scores = (
        linear_obs_model.predict_benchmark_scores_from_capability_scores(
            test_capability_scores
        )
    )

    for b_idx, benchmark in enumerate(benchmarks):
        test_df[f"{benchmark} pred (linear)"] = (
            test_benchmark_scores.T[b_idx].detach().numpy()
        )

    test_df["PC-1 linear"] = test_capability_scores.detach().numpy()

    return linear_obs_model


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
            Spe(f"{benchmark} pred (linear)", "C1"),
        ],
        y_label=benchmark,
    )
    plot_train_test(
        ax_arr[1],
        train,
        test,
        "PC-1 linear",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred (linear)", "C1"),
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
            Spe(f"{benchmark} pred (logit)", "C1"),
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
        "PC-1 logit",
        [
            Spe(f"{benchmark}", "C0"),
            Spe(f"{benchmark} pred (logit)", "C1"),
        ],
        y_label=benchmark,
    )

    plot_train_test(
        ax_arr[3],
        train,
        test,
        "PC-1 logit",
        [
            Spe(f"{benchmark} logit", "C0"),
            Spe(f"{benchmark} logit pred", "C1"),
        ],
        y_label=f"{benchmark} logit",
    )


# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs_splits = list(
    ExpandingWindowBacktestSplitter(
        min_train_size=20, test_size=40, increment=10, key="FLOPs_opt_Besiroglu (1E21)"
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


n_trains = len(ewbs_splits) * len(all_benchmarks)

for split_idx, (train, test) in enumerate(ewbs_splits):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        i_train = split_idx * len(all_benchmarks) + bench_idx
        print(f"Training {i_train}/{n_trains}")
        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, test, benchmark_list)
        logit_model = add_logit_model(train, test, benchmark_list)

        # predict the excluded benchmark
        lin_slaw = add_slaw(train, test, excluded_benchmark, "PC-1 linear")

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # predict the excluded benchmark
        logit_slaw = add_slaw(train, test, excluded_benchmark, "PC-1 logit")

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # store the results
        ewbs_split_train_dict[(split_idx, bench_idx)] = train
        ewbs_split_test_dict[(split_idx, bench_idx)] = test
        ewbs_linear_model_dict[(split_idx, bench_idx)] = linear_model
        ewbs_lin_slaw_dict[(split_idx, bench_idx)] = lin_slaw
        ewbs_lin_slaw_err_dict[(split_idx, bench_idx)] = lin_slaw_err
        ewbs_logit_model_dict[(split_idx, bench_idx)] = logit_model
        ewbs_logit_slaw_dict[(split_idx, bench_idx)] = logit_slaw
        ewbs_logit_slaw_err_dict[(split_idx, bench_idx)] = logit_slaw_err

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

# create plot
fig, ax = plt.subplots(
    len(ewbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(ewbs_splits)),
    squeeze=False,
)

for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        train = ewbs_split_train_dict[(split_idx, bench_idx)]
        test = ewbs_split_test_dict[(split_idx, bench_idx)]
        lin_slaw = ewbs_lin_slaw_dict[(split_idx, bench_idx)]
        lin_slaw_err = ewbs_lin_slaw_err_dict[(split_idx, bench_idx)]
        logit_slaw = ewbs_logit_slaw_dict[(split_idx, bench_idx)]
        logit_slaw_err = ewbs_logit_slaw_err_dict[(split_idx, bench_idx)]
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
                torch.tensor(train["PC-1 linear"].values, dtype=torch.float32)
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
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
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
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
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


# print the mean error
e_err_lin = np.zeros((len(ewbs_splits), len(all_benchmarks)))
e_err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))

for split_idx in range(len(ewbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        e_err_lin[split_idx, bench_idx] = ewbs_lin_slaw_err_dict[(split_idx, bench_idx)]
        e_err_logit[split_idx, bench_idx] = ewbs_logit_slaw_err_dict[
            (split_idx, bench_idx)
        ]

print(f"Expanding Window Mean Linear Error: {e_err_lin.mean()}")
print(f"Expanding Window Mean Logit Error: {e_err_logit.mean()}")

print(
    f"Expanding Window Percent improvement: {100*(e_err_lin.mean() - e_err_logit.mean())/e_err_lin.mean()}"
)

# %%
split_idx = 0
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

    plot_linear_model(ax[bench_idx], bench_idx, train, test, linear_obs_model)

plt.show()


# plot in flop x-space and benchmark y-space
xpoints = train["log10 FLOPs_opt_Besiroglu (1E21)"]
ypoints = train[excluded_benchmark].values
ypoints_2 = (
    lin_slaw.forward(torch.tensor(train["PC-1 linear"].values, dtype=torch.float32))
    .detach()
    .numpy()
)
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
ax[0, 0].set_title(f"{excluded_benchmark} vs FLOPs")
ax[0, 0].set_xlabel("log10 FLOPs (1E21)")
ax[0, 0].set_ylabel(f"{excluded_benchmark}")
ax[0, 0].scatter(xpoints, ypoints, label=excluded_benchmark, marker="x")
ax[0, 0].scatter(
    xpoints, ypoints_2, label=excluded_benchmark + " (observed)", alpha=0.5, marker="x"
)
ax[0, 0].legend()

# plot the test data
xpoints = test["log10 FLOPs_opt_Besiroglu (1E21)"]
ypoints = test[excluded_benchmark].values
ypoints_2 = (
    lin_slaw.forward(torch.tensor(test["PC-1 linear"].values, dtype=torch.float32))
    .detach()
    .numpy()
)
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
ax[0, 0].scatter(xpoints, ypoints, label=excluded_benchmark, marker="o", color="C0")
ax[0, 0].scatter(
    xpoints,
    ypoints_2,
    label=excluded_benchmark + " (observed)",
    alpha=0.5,
    marker="o",
    color="C1",
)
ax[0, 0].legend()


# plot in capability x-space and benchmark y-space
xpoints = train["PC-1 linear"]
ypoints = train[excluded_benchmark].values
ypoints_2 = (
    lin_slaw.forward(torch.tensor(train["PC-1 linear"].values, dtype=torch.float32))
    .detach()
    .numpy()
)
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
ax[0, 1].set_title(f"{excluded_benchmark} vs PC-1")
ax[0, 1].set_xlabel("PC-1")
ax[0, 1].set_ylabel(f"{excluded_benchmark}")
ax[0, 1].scatter(xpoints, ypoints, label=excluded_benchmark, marker="x")
ax[0, 1].scatter(
    xpoints, ypoints_2, label=excluded_benchmark + " (observed)", alpha=0.5, marker="x"
)
ax[0, 1].legend()

# plot the test data
xpoints = test["PC-1 linear"]
ypoints = test[excluded_benchmark].values
ypoints_2 = (
    lin_slaw.forward(torch.tensor(test["PC-1 linear"].values, dtype=torch.float32))
    .detach()
    .numpy()
)
xspace = np.linspace(xpoints.min(), xpoints.max(), 100)
ax[0, 1].scatter(xpoints, ypoints, label=excluded_benchmark, marker="o", color="C0")
ax[0, 1].scatter(
    xpoints,
    ypoints_2,
    label=excluded_benchmark + " (observed)",
    alpha=0.5,
    marker="o",
    color="C1",
)
ax[0, 1].legend()
# %%


split_idx = 0
bench_idx = 0
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
    plot_logit_model(ax[bench_idx], bench_idx, train, test, logit_obs_model)

plt.tight_layout()

plt.show()

# also plot the data for the actual fit curve on the excluded benchmark
fig, ax = plt.subplots(1, 4, figsize=(10, 5), squeeze=False)  # 2 columns
ax_arr = ax[0]
# plot in flop x-space and benchmark y-space

plot_train_test(
    ax_arr[0],
    train,
    test,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred (logit)", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[1],
    train,
    test,
    "log10 FLOPs_opt_Besiroglu (1E21)",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)

plot_train_test(
    ax_arr[2],
    train,
    test,
    "PC-1 logit",
    [
        Spe(f"{excluded_benchmark}", "C0"),
        Spe(f"{excluded_benchmark} pred (logit)", "C1"),
    ],
    y_label=excluded_benchmark,
)

plot_train_test(
    ax_arr[3],
    train,
    test,
    "PC-1 logit",
    [
        Spe(f"{excluded_benchmark} logit", "C0"),
        Spe(f"{excluded_benchmark} logit pred", "C1"),
    ],
    y_label=f"{excluded_benchmark} logit",
)


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
rwbs_lin_slaw_err_dict = {}
rwbs_logit_model_dict = {}
rwbs_logit_slaw_dict = {}
rwbs_logit_slaw_err_dict = {}

for split_idx, (train, test) in enumerate(rwbs_splits):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        benchmark_list = [b for b in all_benchmarks if b != excluded_benchmark]

        linear_model = add_linear_model(train, test, benchmark_list, "PC-1 linear")
        logit_model = add_logit_model(train, test, benchmark_list, "PC-1 logit")

        lin_slaw = ScalingLaw(
            floor=benchmark_floor_dict[excluded_benchmark],
            capability_scores=torch.tensor(
                train["PC-1 linear"].values, dtype=torch.float32
            ),
            benchmark_scores=torch.tensor(
                train[excluded_benchmark].values, dtype=torch.float32
            ),
        )
        lin_slaw.fit()

        # compute error
        lin_slaw_err = F.mse_loss(
            lin_slaw.forward(
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # predict the excluded benchmark
        logit_slaw = ScalingLaw(
            floor=benchmark_floor_dict[excluded_benchmark],
            capability_scores=torch.tensor(
                train["PC-1 logit"].values, dtype=torch.float32
            ),
            benchmark_scores=torch.tensor(
                train[excluded_benchmark].values, dtype=torch.float32
            ),
        )
        logit_slaw.fit()

        # compute error
        logit_slaw_err = F.mse_loss(
            logit_slaw.forward(
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
            ),
            torch.tensor(test[excluded_benchmark].values, dtype=torch.float32),
        ).item()

        # store the results
        rwbs_split_train_dict[(split_idx, bench_idx)] = train
        rwbs_split_test_dict[(split_idx, bench_idx)] = test
        rwbs_linear_model_dict[(split_idx, bench_idx)] = linear_model
        rwbs_lin_slaw_dict[(split_idx, bench_idx)] = lin_slaw
        rwbs_lin_slaw_err_dict[(split_idx, bench_idx)] = lin_slaw_err
        rwbs_logit_model_dict[(split_idx, bench_idx)] = logit_model
        rwbs_logit_slaw_dict[(split_idx, bench_idx)] = logit_slaw
        rwbs_logit_slaw_err_dict[(split_idx, bench_idx)] = logit_slaw_err

# %%

# create plot
fig, ax = plt.subplots(
    len(rwbs_splits),
    len(all_benchmarks),
    figsize=(4 * len(all_benchmarks), 4 * len(rwbs_splits)),
)


for split_idx in range(len(rwbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        train = rwbs_split_train_dict[(split_idx, bench_idx)]
        test = rwbs_split_test_dict[(split_idx, bench_idx)]
        lin_slaw = rwbs_lin_slaw_dict[(split_idx, bench_idx)]
        lin_slaw_err = rwbs_lin_slaw_err_dict[(split_idx, bench_idx)]
        logit_slaw = rwbs_logit_slaw_dict[(split_idx, bench_idx)]
        logit_slaw_err = rwbs_logit_slaw_err_dict[(split_idx, bench_idx)]
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
                torch.tensor(train["PC-1 linear"].values, dtype=torch.float32)
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
                torch.tensor(test["PC-1 linear"].values, dtype=torch.float32)
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
                torch.tensor(test["PC-1 logit"].values, dtype=torch.float32)
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


# print the mean error
r_err_lin = np.zeros((len(rwbs_splits), len(all_benchmarks)))
r_err_logit = np.zeros((len(ewbs_splits), len(all_benchmarks)))

for split_idx in range(len(rwbs_splits)):
    for bench_idx, excluded_benchmark in enumerate(all_benchmarks):
        r_err_lin[split_idx, bench_idx] = rwbs_lin_slaw_err_dict[(split_idx, bench_idx)]
        r_err_logit[split_idx, bench_idx] = rwbs_logit_slaw_err_dict[
            (split_idx, bench_idx)
        ]

print(f"Rolling Window Mean Linear Error: {r_err_lin.mean()}")
print(f"Rolling Window Mean Logit Error: {r_err_logit.mean()}")

print(
    f"Rolling Window Percent improvement: {100*(r_err_lin.mean() - r_err_logit.mean())/r_err_lin.mean()}"
)

#####################################
# Train and fit family-specific linear models of PC-1
#####################################
