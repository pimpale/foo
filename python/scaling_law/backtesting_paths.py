# %%
from abc import abstractmethod
import time
from typing import Any, Type, cast, override
import duckdb
import pandas as pd
import numpy as np
import numpy.typing as npt
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
    BacktestSplitter,
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


def augment_train_test_logit(
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


def augment_train_test_linear(
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


def augment_train_test_slaw(
    slaw: ScalingLaw,
    model: ObsScalingLawPredictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_slaw(slaw, model, train)
    augment_df_slaw(slaw, model, test)


@dataclass
class BacktestDataPoint[T: ObsScalingLawPredictor]:
    split_train: pd.DataFrame
    split_test: pd.DataFrame
    model: T
    slaw: ScalingLaw

    def copy(self):
        """
        Returns a copy of the data point.
        The dataframes are deep copied, and the model and slaw are shallow copied.
        """
        return BacktestDataPoint(
            self.split_train.copy(),
            self.split_test.copy(),
            self.model,
            self.slaw,
        )

@dataclass
class BacktestData:
    splitter_class: Type[BacktestSplitter]
    model_class: Type[ObsScalingLawPredictor]
    results: npt.NDArray[np.object_]


def get_benchmark_list(
    ModelCls: Type[ObsScalingLawPredictor],
    predicted_benchmark: str,
) -> list[str]:
    maybe_fixed_benchmarks = ModelCls.fixed_benchmarks()
    if maybe_fixed_benchmarks is not None:
        benchmark_list = maybe_fixed_benchmarks
    else:
        benchmark_list = ModelCls.necessary_benchmarks() + [
            b for b in all_benchmarks if b != predicted_benchmark
        ]

    return benchmark_list


def backtest_models(
    splitter: BacktestSplitter,
    ModelCls: Type[ObsScalingLawPredictor],
    dataframe: pd.DataFrame,
) -> BacktestData:
    # create object ndarray

    train_test_splits = list(splitter.split(dataframe))

    data = BacktestData(
        splitter_class=type(splitter),
        model_class=ModelCls,
        results=np.empty(
            (len(train_test_splits), len(all_benchmarks)), dtype=np.object_
        ),
    )

    n_trains = len(train_test_splits) * len(all_benchmarks)

    for split_idx, (train, test) in enumerate(train_test_splits):
        for bench_idx, predicted_benchmark in enumerate(all_benchmarks):
            i_train = split_idx * len(all_benchmarks) + bench_idx
            print(f"Training {i_train}/{n_trains}")

            # construct the model inputs
            benchmark_list = get_benchmark_list(ModelCls, predicted_benchmark)

            model_scores = torch.tensor(train[benchmark_list].values, dtype=torch.float32)


            # create model
            model = ModelCls(
                benchmark_list,
                benchmark_floors=[benchmark_floor_dict[b] for b in benchmark_list],
                train_model_scores=model_scores,
            )

            # train
            t0 = time.time()
            model.fit()
            model.eval()
            print(f"Training Time: {time.time() - t0:.2f} seconds")

            # predict the excluded benchmark
            capability_scores = model.predict_capability_scores_from_model_scores(
                model_scores
            ).detach()
            benchmark_scores = torch.tensor(train[predicted_benchmark].values, dtype=torch.float32)
            slaw = ScalingLaw(
                benchmark=predicted_benchmark,
                floor=benchmark_floor_dict[predicted_benchmark],
                capability_scores=capability_scores,
                benchmark_scores=benchmark_scores,
            )
            t0 = time.time()
            slaw.fit()
            slaw.eval()
            print(f"Scaling Law Training Time: {time.time() - t0:.2f} seconds")


            # store the results
            data.results[split_idx, bench_idx] = BacktestDataPoint(
                split_train=train,
                split_test=test,
                model=model,
                slaw=slaw,
            )

    return data


def compute_test_train_error(arr: npt.NDArray[np.object_]) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.float32],
]:
    train_err = np.zeros_like(arr, dtype=np.float32)
    test_err = np.zeros_like(arr, dtype=np.float32)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            bdp: BacktestDataPoint = arr[i, j]
            train = bdp.split_train
            test = bdp.split_test
            slaw = bdp.slaw
            model = bdp.model

            for dataset, dataset_err_arr in ((train, train_err), (test, test_err)):
                x = torch.tensor(dataset[model.benchmarks].values, dtype=torch.float32)
                y = torch.tensor(dataset[slaw.benchmark].values, dtype=torch.float32)
                capability_scores = model.predict_capability_scores_from_model_scores(x)
                y_hat = slaw.forward(capability_scores)
                dataset_err_arr[i, j] = F.mse_loss(
                    y,
                    y_hat,
                ).item()

    return train_err, test_err


# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=40, test_size=20, increment=10, key="FLOPs_opt_Besiroglu (1E21)"
)

ewbs_lin_data = backtest_models(
    ewbs, LinearObsScalingLawPredictor, base_llm_benchmark_eval
)
ewbs_logit_data = backtest_models(
    ewbs, LogitObsScalingLawPredictor, base_llm_benchmark_eval
)

# %%

######
# Compute the mean error of the linear and logit models
######

ewbs_lin_train_err, ewbs_lin_test_err = compute_test_train_error(ewbs_lin_data.results)
ewbs_logit_train_err, ewbs_logit_test_err = compute_test_train_error(ewbs_logit_data.results)

print(f"Linear Train Error: {ewbs_lin_train_err.mean()}")
print(f"Logit Train Error: {ewbs_logit_train_err.mean()}")
print(f"Linear Test Error: {ewbs_lin_test_err.mean()}")
print(f"Logit Test Error: {ewbs_logit_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_lin_train_err.mean() - ewbs_logit_train_err.mean()) / ewbs_lin_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_lin_test_err.mean() - ewbs_logit_test_err.mean()) / ewbs_lin_test_err.mean() * 100:.2f}%"
)


# %%

def plot_linear_scaling_law(lin_data_point: BacktestDataPoint[LinearObsScalingLawPredictor]):
    fig, ax = plt.subplots(
        len(lin_data_point.model.benchmarks),
        2,
        figsize=(10, len(lin_data_point.model.benchmarks) * 5),
        squeeze=False,
    )  # 1 columns

    # insert data from excluded benchmark

    for bench_idx, benchmark in enumerate(lin_data_point.model.benchmarks):
        pt = lin_data_point.copy()
        augment_train_test_linear(pt.model, pt.split_train, pt.split_test)
        plot_linear_model(ax[bench_idx], bench_idx, pt.split_train, pt.split_test, pt.model)

    # now plot the data for the actual fit curve on the excluded benchmark
    # 1 row, 4 columns
    # col 0: FLOPs vs benchmark (show both true and predicted)
    # col 1: FLOPs vs logit benchmark (show both true and predicted)
    # col 2: capability vs benchmark (show both true and predicted)
    # col 3: capability vs logit benchmark (show both true and predicted)

    pt = lin_data_point.copy()
    augment_train_test_slaw(pt.slaw, pt.model, pt.split_train, pt.split_test)
    excluded_benchmark = pt.slaw.benchmark

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
    ax_arr = ax[0]
    # plot in flop x-space and benchmark y-space
    plot_train_test(
        ax_arr[0],
        pt.split_train,
        pt.split_test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{excluded_benchmark}", "C0"),
            Spe(f"{excluded_benchmark} pred", "C1"),
        ],
        y_label=excluded_benchmark,
    )

    plot_train_test(
        ax_arr[1],
        pt.split_train,
        pt.split_test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{excluded_benchmark} logit", "C0"),
            Spe(f"{excluded_benchmark} logit pred", "C1"),
        ],
        y_label=f"{excluded_benchmark} logit",
    )

    plot_train_test(
        ax_arr[2],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{excluded_benchmark}", "C0"),
            Spe(f"{excluded_benchmark} pred", "C1"),
        ],
        y_label=excluded_benchmark,
    )

    plot_train_test(
        ax_arr[3],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{excluded_benchmark} logit", "C0"),
            Spe(f"{excluded_benchmark} logit pred", "C1"),
        ],
        y_label=f"{excluded_benchmark} logit",
    )

    plt.show()

split_idx = 0
bench_idx = 1
plot_linear_scaling_law(ewbs_lin_data.results[split_idx, bench_idx])
# %%

def plot_logit_scaling_law(logit_data_point: BacktestDataPoint[LogitObsScalingLawPredictor]):
    fig, ax = plt.subplots(
        len(logit_data_point.model.benchmarks),
        4,
        figsize=(4 * 4, len(logit_data_point.model.benchmarks) * 4),
        squeeze=False,
    )  # 1 columns

    for bench_idx, benchmark in enumerate(logit_data_point.model.benchmarks):
        pt = logit_data_point.copy()
        augment_train_test_logit(pt.model, pt.split_train,pt.split_test)
        plot_logit_model(ax[bench_idx], bench_idx, pt.split_train, pt.split_test, pt.model)

    plt.tight_layout()

    plt.show()

    pt = logit_data_point.copy()
    augment_train_test_slaw(logit_slaw, pt.model, pt.split_train, pt.split_test)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
    ax_arr = ax[0]
    # plot in flop x-space and benchmark y-space
    plot_train_test(
        ax_arr[0],
        pt.split_train,
        pt.split_test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{excluded_benchmark}", "C0"),
            Spe(f"{excluded_benchmark} pred", "C1"),
        ],
        y_label=excluded_benchmark,
    )

    plot_train_test(
        ax_arr[1],
        pt.split_train,
        pt.split_test,
        "log10 FLOPs_opt_Besiroglu (1E21)",
        [
            Spe(f"{excluded_benchmark} logit", "C0"),
            Spe(f"{excluded_benchmark} logit pred", "C1"),
        ],
        y_label=f"{excluded_benchmark} logit",
    )

    plot_train_test(
        ax_arr[2],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{excluded_benchmark}", "C0"),
            Spe(f"{excluded_benchmark} pred", "C1"),
        ],
        y_label=excluded_benchmark,
    )

    plot_train_test(
        ax_arr[3],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{excluded_benchmark} logit", "C0"),
            Spe(f"{excluded_benchmark} logit pred", "C1"),
        ],
        y_label=f"{excluded_benchmark} logit",
    )

    plt.show()

split_idx = 0
bench_idx = 1
plot_logit_scaling_law(ewbs_logit_data.results[split_idx, bench_idx])

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

rwbs_data = backtest_models(rwbs_splits)

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
        augment_train_test_linear(linear_model, train, test)
        augment_train_test_logit(logit_model, train, test)

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
