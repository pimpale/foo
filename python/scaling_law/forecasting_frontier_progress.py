# %%
from collections import defaultdict
import time
from typing import Any, Type, cast, override
import duckdb
import pandas as pd
import numpy as np
import numpy.typing as npt
import torch
import torch._dynamo.cache_size
import torch.nn.functional as F
from dataclasses import dataclass
import matplotlib.pyplot as plt
import matplotlib.axes
from tqdm import tqdm
import seaborn as sns
import ipyvolume as ipv

from util_frontier import Frontier, get_running_top_n
from util_obs_scaling_law_predictor import ScalingLaw
from util_timeseries_backtesting import (
    BacktestSplitter,
    ExpandingWindowBacktestSplitter,
)

from util_frontier_date_predictor import FrontierDatePredictor
from util_frontier_flop_predictor import FrontierFlopPredictor
from util_frontier_flop_date_predictor import FrontierFlopDatePredictor
from util_frontier_flop_to_elo_predictor import FrontierFlopToEloPredictor
from util_frontier_date_to_elo_predictor import FrontierDateToEloPredictor
from util_frontier_flop_to_pc1_predictor import FrontierFlopToPC1Predictor
from util_frontier_date_to_pc1_predictor import FrontierDateToPC1Predictor

torch.set_num_threads(1)
torch._dynamo.cache_size.config.cache_size_limit = 1e9

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
        "Model" as model,
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
        "chatbot_arena_name" as model,
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

agentic_benchmark = duckdb.read_csv("./data_models/cache_new/agentic_benchmark.csv")
agentic_benchmark = duckdb.sql(
    """
    SELECT
        "Model" as model,
        "Chatbot Arena Elo" as Elo,
        "SWE-Bench Verified",
        "Cybench",
        "RE-Bench total",
        year("Release Date") + (1/365)*dayofyear("Release Date") as release_date
    FROM agentic_benchmark
    UNION ALL VALUES
        ('a', 0, 0, 0, 0, 2025.00),
        ('b', 0, 0, 0, 0, 2025.25),
        ('c', 0, 0, 0, 0, 2025.50),
        ('d', 0, 0, 0, 0, 2025.75),
        ('e', 0, 0, 0, 0, 2026.00)
    """
).df()
agentic_benchmark_benchmarks = [
    "SWE-Bench Verified",
    "Cybench",
    "RE-Bench total"
]


# drop NaNs
base_llm_benchmark_eval.dropna(inplace=True)
openllm_elo_merged.dropna(inplace=True)
# agentic_benchmark.dropna(inplace=True)

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
    ("SWE-Bench Verified", 0),
    ("Cybench", 0),
    ("RE-Bench total", 0),
]
benchmark_floor_dict = defaultdict(lambda: 0.0, {b: f for b, f in benchmark_data})


@dataclass
class Spe:
    """
    Scatter Plot Entry
    """

    y_key: str
    y_label: str
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
            marker="x",
            alpha=0.5,
            color=e.color,
        )
        ax.scatter(
            test_df[x_key],
            test_df[e.y_key],
            marker="o",
            label=e.y_label,
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


def augment_df_slaw(model: Frontier, df_to_augment: pd.DataFrame):
    model_scores = torch.tensor(
        df_to_augment[model.benchmarks].values, dtype=torch.float32
    )
    df_to_augment[f"{model.slaw.benchmark} pred"] = (
        model.predict_benchmark_scores(model_scores).detach().numpy()
    )


def augment_train_test_slaw(
    model: Frontier,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_slaw(model, train)
    augment_df_slaw(model, test)


@dataclass
class BacktestDataPoint[T: Frontier]:
    split_train: pd.DataFrame
    split_test: pd.DataFrame
    model: T

    def copy(self):
        """
        Returns a copy of the data point.
        The dataframes are deep copied, and the model and slaw are shallow copied.
        """
        return BacktestDataPoint(
            self.split_train.copy(),
            self.split_test.copy(),
            self.model,
        )



@dataclass
class BacktestFrontierData:
    splitter: BacktestSplitter
    model_class: Type[Frontier]
    benchmarks: list[str]
    splits: list[str]
    # 2D array of BacktestDataPoint on the splits x benchmarks
    results: npt.NDArray[np.object_]
    # 1D array of BacktestDataPoint on the benchmarks (using all points)
    global_split_results: npt.NDArray[np.object_]

def get_benchmark_list(
    ModelCls: Type[Frontier],
    predicted_benchmark: str,
    dataframe_benchmarks: list[str],
) -> list[str]:
    return ModelCls.necessary_benchmarks() + [
        b for b in dataframe_benchmarks if b != predicted_benchmark
    ]



def backtest_models_frontier(
    splitter: BacktestSplitter,
    ModelCls: Type[Frontier],
    dataframe: pd.DataFrame,
    dataframe_benchmarks: list[str],
) -> BacktestFrontierData:
    # create object ndarray

    train_test_splits = list(splitter.split(dataframe))

    data = BacktestFrontierData(
        splitter=splitter,
        model_class=ModelCls,
        benchmarks=dataframe_benchmarks,
        splits=[f"split_{i}" for i in range(len(train_test_splits))],
        results=np.empty(
            (len(train_test_splits), len(dataframe_benchmarks)), dtype=np.object_
        ),
        global_split_results=np.empty(len(dataframe_benchmarks), dtype=np.object_),
    )

    n_trains = (len(train_test_splits) + 1) * len(dataframe_benchmarks)

    for split_idx, (train, test) in enumerate(
        [(dataframe, dataframe.head(0))] + train_test_splits
    ):
        for bench_idx, predicted_benchmark in enumerate(dataframe_benchmarks):
            i_train = split_idx * len(dataframe_benchmarks) + bench_idx
            print(f"Training {i_train}/{n_trains}")

            # construct the model inputs
            benchmark_list = get_benchmark_list(
                ModelCls, predicted_benchmark, dataframe_benchmarks
            )

            t0 = time.time()

            # create model
            model = ModelCls(
                benchmark_list,
                benchmark_floors=[benchmark_floor_dict[b] for b in benchmark_list],
                target_benchmark=predicted_benchmark,
                target_benchmark_floor=benchmark_floor_dict[predicted_benchmark],
                train_df=train,
            )

            # train
            model.fit()
            model.eval()
            print(f"{ModelCls.__name__} Training Time: {time.time() - t0:.2f} seconds")

            # store the results
            if split_idx == 0:
                data.global_split_results[bench_idx] = BacktestDataPoint(
                    split_train=train,
                    split_test=test,
                    model=model,
                )
            else:
                data.results[split_idx - 1, bench_idx] = BacktestDataPoint(
                    split_train=train,
                    split_test=test,
                    model=model,
                )

    return data



def compute_test_train_error_frontier(arr: npt.NDArray[np.object_]) -> tuple[
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
            model = bdp.model

            for dataset, dataset_err_arr in ((train, train_err), (test, test_err)):
                dataset_frontier = get_running_top_n(
                    dataset, "release_date", model.slaw.benchmark, 3, "model"
                )

                x = torch.tensor(
                    dataset_frontier[model.benchmarks].values, dtype=torch.float32
                )
                y = torch.tensor(
                    dataset_frontier[model.slaw.benchmark].values, dtype=torch.float32
                )
                y_hat = model.predict_benchmark_scores(x)
                dataset_err_arr[i, j] = F.mse_loss(
                    y,
                    y_hat,
                ).item()

    return train_err, test_err


def plot_comparison(backtests: list[BacktestFrontierData], expand=False):
    assert len(backtests) > 0
    b0 = backtests[0]
    n_split, n_bench = b0.results.shape

    # key on which we split
    x_key = b0.splitter.key

    if expand:
        fig, ax = plt.subplots(
            n_split * len(backtests),
            n_bench,
            figsize=(4 * n_bench, 4 * n_split * len(backtests)),
            squeeze=False,
        )
    else:
        fig, ax = plt.subplots(
            n_split,
            n_bench,
            figsize=(4 * n_bench, 4 * n_split),
            squeeze=False,
        )

    for i, b in enumerate(backtests):
        # plot ground truth data
        for split_idx in range(n_split):
            if expand:
                y_idx = split_idx * len(backtests) + i
            else:
                y_idx = split_idx
            for bench_idx in range(n_bench):
                if i == 0 or expand:
                    b0dp: BacktestDataPoint = b0.results[split_idx, bench_idx]
                    # We plot the ground truth data
                    plot_train_test(
                        ax[y_idx, bench_idx],
                        b0dp.split_train,
                        b0dp.split_test,
                        x_key,
                        [Spe(b0dp.model.slaw.benchmark, "Ground Truth", "black")],
                        y_label=b0dp.model.slaw.benchmark,
                    )

                if expand:
                    color = "C0"
                else:
                    color = f"C{i}"

                # otherwise plot the model data
                bdp: BacktestDataPoint = b.results[split_idx, bench_idx]
                bdp_copy = bdp.copy()
                augment_train_test_slaw(
                    bdp_copy.model,
                    bdp_copy.split_train,
                    bdp_copy.split_test,
                )
                plot_train_test(
                    ax[y_idx, bench_idx],
                    bdp_copy.split_train,
                    bdp_copy.split_test,
                    x_key,
                    [
                        Spe(
                            f"{bdp_copy.model.slaw.benchmark} pred",
                            f"{type(bdp_copy.model).__name__} pred",
                            color,
                        ),
                    ],
                    y_label=bdp_copy.model.slaw.benchmark,
                )

    fig.tight_layout()
    plt.show()



def plot_split(
    backtest: BacktestFrontierData,
    benchmark_id: int,
    x_key: str,
    expand=False,
    line=False,
    capability=False,
):

    color_list = [
        "tab:blue",
        "tab:cyan",
        "tab:green",
        "tab:orange",
    ]

    n_split, n_bench = backtest.results.shape
    assert benchmark_id < n_bench

    assert not (line and x_key != backtest.splitter.key), "Cannot plot line without x_key being the split key"

    if expand:
        fig, ax = plt.subplots(1, n_split, figsize=(5 * n_split, 5), squeeze=False)
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)

    # first, plot the train points.
    # We need to use the final dataframe to get the ground truth data, since it will have the best-trained PC-1 score (and doesn't matter for the other models)
    bdp_g: BacktestDataPoint = backtest.global_split_results[benchmark_id]
    bdp_g_copy = bdp_g.copy()
    print(bdp_g_copy.model)
    
    augment_df_slaw(
        bdp_g_copy.model,
        bdp_g_copy.split_train,
    )

    bdp_g_splits = list(backtest.splitter.split(bdp_g_copy.split_train))

    frontier_set = set(
        get_running_top_n(
            bdp_g_copy.split_train,
            backtest.splitter.key,
            bdp_g.model.slaw.benchmark,
            1,
            "model",
        )["model"]
    )
    

    last_max_v = bdp_g_copy.split_train[backtest.splitter.key].min()

    for j in range(len(bdp_g_splits) if expand else 1):
        curr_ax = ax[0, j]

        plotted_points = set()

        for i, (train, test) in enumerate(bdp_g_splits):
            max_v = train[backtest.splitter.key].max()
            min_v = last_max_v
            last_max_v = max_v

            df = train[~train[backtest.splitter.key].isin(plotted_points)]

            if capability:
                y_key = f"{bdp_g.model.slaw.benchmark} capability score"
            else:
                y_key = f"{bdp_g.model.slaw.benchmark}"

            curr_ax.scatter(
                df[x_key],
                df[y_key],
                label=f"{min_v:.1f} - {max_v:.1f} {backtest.splitter.key}",
                alpha=[1 if m in frontier_set else 0.75 for m in df["model"]],
                s=[40 if m in frontier_set else 20 for m in df["model"]],
                color=color_list[i],
            )
            curr_ax.set_title(f"{x_key} vs {y_key}")
            curr_ax.set_xlabel(x_key)
            curr_ax.set_ylabel(bdp_g.model.slaw.benchmark)

            plotted_points.update(train[backtest.splitter.key])

            if i == len(bdp_g_splits) - 1:
                df = bdp_g_copy.split_train[
                    ~bdp_g_copy.split_train[backtest.splitter.key].isin(plotted_points)
                ]
                curr_ax.scatter(
                    df[x_key],
                    df[y_key],
                    label=f"{max_v:.1f} + {backtest.splitter.key}",
                    alpha=[1 if m in frontier_set else 0.75 for m in df["model"]],
                    s=[40 if m in frontier_set else 20 for m in df["model"]],
                    color=color_list[len(bdp_g_splits)],
                )
                curr_ax.set_title(f"{x_key} vs {y_key}")
                curr_ax.set_xlabel(x_key)
                curr_ax.set_ylabel(y_key)

    # now plot the predictions
    # to do this, we use the model to make predictions for the entire space and plot it

    for split_idx in range(n_split):
        color = color_list[split_idx]
        bdp: BacktestDataPoint[Frontier] = backtest.results[split_idx, benchmark_id]

        # augment the global split with the model's predictions
        bdp_g_copy2 = bdp_g.copy()
        augment_df_slaw(bdp.model, bdp_g_copy2.split_train)

        if expand:
            curr_ax = ax[0, split_idx]
        else:
            curr_ax = ax[0, 0]

        if capability:
            label = f"{type(bdp.model).__name__} capability"
            y_key = f"{bdp.model.slaw.benchmark} pred capability score"
        else:
            label = f"{type(bdp.model).__name__} pred"
            y_key = f"{bdp.model.slaw.benchmark} pred"

        # plot the predictions
        if line:
            xs = np.array(bdp_g_copy.split_train[x_key])
            ys = np.array(bdp_g_copy2.split_train[y_key])

            # Sort both arrays based on x values
            sort_idx = np.argsort(xs)

            curr_ax.plot(
                xs[sort_idx],
                ys[sort_idx],
                label=label,
                alpha=1,
                color=color,
            )
        else:
            curr_ax.scatter(
                bdp_g_copy.split_train[x_key],
                bdp_g_copy2.split_train[y_key],
                label=label,
                alpha=1,
                marker="x",
                color=color,
            )

        min_v = bdp_g_copy.split_train[backtest.splitter.key].min()
        max_v = bdp_g_copy.split_train[backtest.splitter.key].max()

        curr_ax.legend()

    fig.tight_layout()
    plt.show()


def plot_errmatrix_comparison(
    backtests: list[BacktestFrontierData],
):
    assert len(backtests) > 0
    methods = [b.model_class.__name__.replace("Predictor", "") for b in backtests]
    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(2, 3, figsize=(30 * 0.7, 20 * 0.7))

    # create 3d matrix of errors
    train_errs = np.zeros(
        (
            # methods
            len(backtests),
            # splits
            len(backtests[0].splits),
            # benchmarks
            len(backtests[0].benchmarks),
        )
    )
    test_errs = np.zeros_like(train_errs)  # same shape as err_train

    for i, b in enumerate(backtests):
        train_err, test_err = compute_test_train_error_frontier(b.results)
        train_errs[i] = train_err
        test_errs[i] = test_err

    train_vmax = np.max(np.sqrt(train_errs)).item()
    test_vmax = np.max(np.sqrt(test_errs)).item()

    # aggregate over splits
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=1).T),
        ax=ax[0, 0],
        yticklabels=backtests[0].benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=1).T),
        ax=ax[1, 0],
        yticklabels=backtests[0].benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    # aggregate over benchmarks
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=2).T),
        ax=ax[0, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=2).T),
        ax=ax[1, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax,
    )

    # aggregate over methods
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=0).T),
        ax=ax[0, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        annot=True,
        vmin=0,
        vmax=train_vmax,
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=0).T),
        ax=ax[1, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        vmin=0,
        vmax=test_vmax,
        annot=True,
    )

    # set column titles
    ax[0, 0].set_title("Predictor perf on Benchmark", size="xx-large")
    ax[0, 1].set_title("Predictor perf on Split", size="xx-large")
    ax[0, 2].set_title("Overall perf on (Split, Benchmark)", size="xx-large")

    # set row titles
    ax[0, 0].set_ylabel("Train Set", size="xx-large")
    ax[1, 0].set_ylabel("Test Set", size="xx-large")

    fig.tight_layout()
    plt.show()


def plot_all_loss_curves(data: BacktestFrontierData):
    n_split, n_bench = data.results.shape
    fig, ax = plt.subplots(
        n_split,
        n_bench,
        figsize=(4 * n_bench, 4 * n_split),
        squeeze=False,
    )
    for split_idx in range(n_split):
        for bench_idx in range(n_bench):
            bdp: BacktestDataPoint = data.results[split_idx, bench_idx]
            slaw = bdp.model.slaw
            ax[split_idx, bench_idx].plot(np.log10(slaw.train_losses[:]), label="train")
            ax[split_idx, bench_idx].set_title(slaw.benchmark)
            ax[split_idx, bench_idx].legend()


def compare(
    data1: BacktestFrontierData,
    data2: BacktestFrontierData,
):
    plot_errmatrix_comparison([data1, data2])

    print(f"{data1.model_class.__name__} Train Error: {data1.results.mean()}")
    print(f"{data2.model_class.__name__} Test Error: {data2.results.mean()}")

    print(
        f"Train Percentage Improvement: {(data1.results.mean() - data2.results.mean()) / data1.results.mean() * 100:.2f}%"
    )
    print(
        f"Test Percentage Improvement: {(data1.results.mean() - data2.results.mean()) / data1.results.mean() * 100:.2f}%"
    )

#%%

ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=7,
    test_size=4,
    increment=10,
    key="release_date",
)

#%%
ewbs_frontier_date_to_elo_data = backtest_models_frontier(
    ewbs, FrontierDateToEloPredictor, agentic_benchmark, agentic_benchmark_benchmarks
)

ewbs_frontier_date_to_elo_train_err, ewbs_frontier_date_to_elo_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_to_elo_data.results)
)

#%%
plot_comparison(
    [
        ewbs_frontier_date_to_elo_data,
    ]
)

#%%
plot_split(
    ewbs_frontier_date_to_elo_data,
    2,
    "release_date",
    expand=False,
    line=True,
    capability=False,
)

# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=10,
    test_size=10,
    increment=10,
    key="release_date",
)


# %%
ewbs_frontier_date_data = backtest_models_frontier(
    ewbs, FrontierDatePredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_train_err, ewbs_frontier_date_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_data.results)
)
# %%
ewbs_frontier_flop_data = backtest_models_frontier(
    ewbs, FrontierFlopPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_train_err, ewbs_frontier_flop_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_data.results)
)

# %%
ewbs_frontier_flop_date_data = backtest_models_frontier(
    ewbs, FrontierFlopDatePredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_date_train_err, ewbs_frontier_flop_date_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_date_data.results)
)

# %%
ewbs_frontier_flop_to_elo_data = backtest_models_frontier(
    ewbs, FrontierFlopToEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_to_elo_train_err, ewbs_frontier_flop_to_elo_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_to_elo_data.results)
)

# %%
ewbs_frontier_date_to_elo_data = backtest_models_frontier(
    ewbs, FrontierDateToEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_to_elo_train_err, ewbs_frontier_date_to_elo_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_to_elo_data.results)
)

# %%
ewbs_frontier_flop_to_pc1_data = backtest_models_frontier(
    ewbs, FrontierFlopToPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_flop_to_pc1_train_err, ewbs_frontier_flop_to_pc1_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_flop_to_pc1_data.results)
)

# %%
ewbs_frontier_date_to_pc1_data = backtest_models_frontier(
    ewbs, FrontierDateToPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_frontier_date_to_pc1_train_err, ewbs_frontier_date_to_pc1_test_err = (
    compute_test_train_error_frontier(ewbs_frontier_date_to_pc1_data.results)
)

# %%

# print ALL of the average errors:
# Date, Flop, FlopDate, FlopToElo, DateToElo, FlopToPC1, DateToPC1

print(f"Date -> Downstream Train MSE: {ewbs_frontier_date_train_err.mean():.3f}")
print(f"Flop -> Downstream Train MSE: {ewbs_frontier_flop_train_err.mean():.3f}")
print(
    f"FlopDate -> Downstream Train MSE: {ewbs_frontier_flop_date_train_err.mean():.3f}"
)
print(
    f"FlopToElo -> Downstream Train MSE: {ewbs_frontier_flop_to_elo_train_err.mean():.3f}"
)
print(
    f"DateToElo -> Downstream Train MSE: {ewbs_frontier_date_to_elo_train_err.mean():.3f}"
)
print(
    f"FlopToPC1 -> Downstream Train MSE: {ewbs_frontier_flop_to_pc1_train_err.mean():.3f}"
)
print(
    f"DateToPC1 -> Downstream Train MSE: {ewbs_frontier_date_to_pc1_train_err.mean():.3f}"
)

print()

print(f"Date -> Downstream Test MSE: {ewbs_frontier_date_test_err.mean():.3f}")
print(f"Flop -> Downstream Test MSE: {ewbs_frontier_flop_test_err.mean():.3f}")

print(f"FlopDate -> Downstream Test MSE: {ewbs_frontier_flop_date_test_err.mean():.3f}")

print(
    f"FlopToElo -> Downstream Test MSE: {ewbs_frontier_flop_to_elo_test_err.mean():.3f}"
)
print(
    f"DateToElo -> Downstream Test MSE: {ewbs_frontier_date_to_elo_test_err.mean():.3f}"
)
print(
    f"FlopToPC1 -> Downstream Test MSE: {ewbs_frontier_flop_to_pc1_test_err.mean():.3f}"
)
print(
    f"DateToPC1 -> Downstream Test MSE: {ewbs_frontier_date_to_pc1_test_err.mean():.3f}"
)

# print RMSE
print()
print()

print(f"Date -> Downstream Train RMSE: {ewbs_frontier_date_train_err.mean()**0.5:.3f}")
print(f"Flop -> Downstream Train RMSE: {ewbs_frontier_flop_train_err.mean()**0.5:.3f}")
print(
    f"FlopDate -> Downstream Train RMSE: {ewbs_frontier_flop_date_train_err.mean()**0.5:.3f}"
)
print(
    f"FlopToElo -> Downstream Train RMSE: {ewbs_frontier_flop_to_elo_train_err.mean()**0.5:.3f}"
)
print(
    f"DateToElo -> Downstream Train RMSE: {ewbs_frontier_date_to_elo_train_err.mean()**0.5:.3f}"
)
print(
    f"FlopToPC1 -> Downstream Train RMSE: {ewbs_frontier_flop_to_pc1_train_err.mean()**0.5:.3f}"
)
print(
    f"DateToPC1 -> Downstream Train RMSE: {ewbs_frontier_date_to_pc1_train_err.mean()**0.5:.3f}"
)

print()

print(f"Date -> Downstream Test RMSE: {ewbs_frontier_date_test_err.mean()**0.5:.3f}")
print(f"Flop -> Downstream Test RMSE: {ewbs_frontier_flop_test_err.mean()**0.5:.3f}")

print(
    f"FlopDate -> Downstream Test RMSE: {ewbs_frontier_flop_date_test_err.mean()**0.5:.3f}"
)

print(
    f"FlopToElo -> Downstream Test RMSE: {ewbs_frontier_flop_to_elo_test_err.mean()**0.5:.3f}"
)
print(
    f"DateToElo -> Downstream Test RMSE: {ewbs_frontier_date_to_elo_test_err.mean()**0.5:.3f}"
)
print(
    f"FlopToPC1 -> Downstream Test RMSE: {ewbs_frontier_flop_to_pc1_test_err.mean()**0.5:.3f}"
)
print(
    f"DateToPC1 -> Downstream Test RMSE: {ewbs_frontier_date_to_pc1_test_err.mean()**0.5:.3f}"
)

# %%
plot_comparison(
    [
        ewbs_frontier_date_to_pc1_data,
    ]
)

# %%

# compare the models

plot_comparison(
    [
        ewbs_frontier_date_data,
        ewbs_frontier_flop_data,
        ewbs_frontier_flop_date_data,
        ewbs_frontier_flop_to_elo_data,
        ewbs_frontier_date_to_elo_data,
        ewbs_frontier_flop_to_pc1_data,
        ewbs_frontier_date_to_pc1_data,
    ],
    expand=True,
)


# %%

plot_errmatrix_comparison(
    [
        ewbs_frontier_date_data,
        ewbs_frontier_flop_data,
        ewbs_frontier_flop_date_data,
        ewbs_frontier_flop_to_elo_data,
        ewbs_frontier_date_to_elo_data,
        ewbs_frontier_flop_to_pc1_data,
        ewbs_frontier_date_to_pc1_data,
    ]
)