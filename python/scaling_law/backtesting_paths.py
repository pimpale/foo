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

from util_algprog_logflop_predictor import AlgprogLogFlopPredictor
from util_direct_elo_predictor import DirectEloPredictor
from util_direct_flop_predictor import DirectLogFlopPredictor
from util_flop_date_predictor import LogFlopDatePredictor
from util_obs_scaling_law_predictor import ObsScalingLawPredictor, ScalingLaw
from util_timeseries_backtesting import (
    BacktestSplitter,
    ExpandingWindowBacktestSplitter,
    RollingWindowBacktestSplitter,
)
from util_linear_obs_scaling_law_predictor import LinearPC1Predictor
from util_logit_obs_scaling_law_predictor import LogitPC1Predictor

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


def augment_df_logit(logit_obs_model: LogitPC1Predictor, df_to_augment: pd.DataFrame):
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
    logit_obs_model: LogitPC1Predictor,
    train: pd.DataFrame,
    test: pd.DataFrame,
):
    augment_df_logit(logit_obs_model, train)
    augment_df_logit(logit_obs_model, test)


def augment_df_linear(
    linear_obs_model: LinearPC1Predictor, df_to_augment: pd.DataFrame
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
    linear_obs_model: LinearPC1Predictor,
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
    splitter: BacktestSplitter
    model_class: Type[ObsScalingLawPredictor]
    benchmarks: list[str]
    splits: list[str]
    results: npt.NDArray[np.object_]


def get_benchmark_list(
    ModelCls: Type[ObsScalingLawPredictor],
    predicted_benchmark: str,
    dataframe_benchmarks: list[str],
) -> list[str]:
    maybe_fixed_benchmarks = ModelCls.fixed_benchmarks()
    if maybe_fixed_benchmarks is not None:
        benchmark_list = maybe_fixed_benchmarks
    else:
        benchmark_list = ModelCls.necessary_benchmarks() + [
            b for b in dataframe_benchmarks if b != predicted_benchmark
        ]

    return benchmark_list


def backtest_models(
    splitter: BacktestSplitter,
    ModelCls: Type[ObsScalingLawPredictor],
    dataframe: pd.DataFrame,
    dataframe_benchmarks: list[str],
) -> BacktestData:
    # create object ndarray

    train_test_splits = list(splitter.split(dataframe))

    data = BacktestData(
        splitter=splitter,
        model_class=ModelCls,
        benchmarks=dataframe_benchmarks,
        splits=[f"split_{i}" for i in range(len(train_test_splits))],
        results=np.empty(
            (len(train_test_splits), len(dataframe_benchmarks)), dtype=np.object_
        ),
    )

    n_trains = len(train_test_splits) * len(dataframe_benchmarks)

    for split_idx, (train, test) in enumerate(train_test_splits):
        for bench_idx, predicted_benchmark in enumerate(dataframe_benchmarks):
            i_train = split_idx * len(dataframe_benchmarks) + bench_idx
            print(f"Training {i_train}/{n_trains}")

            # construct the model inputs
            benchmark_list = get_benchmark_list(
                ModelCls, predicted_benchmark, dataframe_benchmarks
            )

            model_scores = torch.tensor(
                train[benchmark_list].values, dtype=torch.float32
            )

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
            print(f"{ModelCls.__name__} Training Time: {time.time() - t0:.2f} seconds")

            # predict the excluded benchmark
            capability_scores = model.predict_capability_scores_from_model_scores(
                model_scores
            ).detach()
            benchmark_scores = torch.tensor(
                train[predicted_benchmark].values, dtype=torch.float32
            )
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


def plot_comparison(backtests: list[BacktestData], expand=False):
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
                        [Spe(b0dp.slaw.benchmark, "Ground Truth", "black")],
                        y_label=b0dp.slaw.benchmark,
                    )

                if expand:
                    color = "C0"
                else:
                    color = f"C{i}"

                # otherwise plot the model data
                bdp: BacktestDataPoint = b.results[split_idx, bench_idx]
                bdp_copy = bdp.copy()
                augment_train_test_slaw(
                    bdp_copy.slaw,
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
                            f"{bdp_copy.slaw.benchmark} pred",
                            f"{type(bdp_copy.model).__name__} pred",
                            color,
                        ),
                    ],
                    y_label=bdp_copy.slaw.benchmark,
                )

    fig.tight_layout()
    plt.show()


def plot_errmatrix_comparison(
    backtests: list[BacktestData],
):
    assert len(backtests) > 0
    methods = [b.model_class.__name__.replace("Predictor", "") for b in backtests]
    # create 3 graphs for each split in [test, train]:
    # 1. Aggregate over benchmarks
    # 2. Aggregate over splits
    # 3. Aggregate over both
    fig, ax = plt.subplots(2, 3, figsize=(30*0.7, 20*0.7))

    # create 3d matrix of errors
    train_errs = np.zeros(
        (
            # methods
            len(backtests), 
            # splits
            len(backtests[0].splits), 
            # benchmarks
            len(backtests[0].benchmarks))
    )
    test_errs = np.zeros_like(train_errs) # same shape as err_train
    
    for i, b in enumerate(backtests):
        train_err, test_err = compute_test_train_error(b.results)
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
        vmax=train_vmax
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=1).T),
        ax=ax[1, 0],
        yticklabels=backtests[0].benchmarks,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax
    )

    # aggregate over benchmarks
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=2).T),
        ax=ax[0, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=train_vmax
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=2).T),
        ax=ax[1, 1],
        yticklabels=backtests[0].splits,
        xticklabels=methods,
        annot=True,
        vmin=0,
        vmax=test_vmax
    )

    # aggregate over methods
    sns.heatmap(
        np.sqrt(train_errs.mean(axis=0).T),
        ax=ax[0, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        annot=True,
        vmin=0,
        vmax=train_vmax
    )
    sns.heatmap(
        np.sqrt(test_errs.mean(axis=0).T),
        ax=ax[1, 2],
        yticklabels=backtests[0].benchmarks,
        xticklabels=backtests[0].splits,
        vmin=0,
        vmax=test_vmax,
        annot=True
    )
    
    
    # set column titles
    ax[0, 0].set_title("Predictor perf on Benchmark", size='xx-large')
    ax[0, 1].set_title("Predictor perf on Split", size='xx-large')
    ax[0, 2].set_title("Overall perf on (Split, Benchmark)", size='xx-large')

    # set row titles
    ax[0, 0].set_ylabel("Train Set", size='xx-large')
    ax[1, 0].set_ylabel("Test Set", size='xx-large')

    fig.tight_layout()
    plt.show()

def plot_all_loss_curves(data: BacktestData):
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
            slaw = bdp.slaw
            ax[split_idx, bench_idx].plot(
                np.log10(slaw.train_losses[500:]), label="train"
            )
            ax[split_idx, bench_idx].set_title(slaw.benchmark)
            ax[split_idx, bench_idx].legend()


def plot_slaw[
    T: ObsScalingLawPredictor
](point: BacktestDataPoint[T],):
    # now plot the data for the actual fit curve on the excluded benchmark
    # 1 row, 4 columns
    # col 0: FLOPs vs benchmark (show both true and predicted)
    # col 1: FLOPs vs logit benchmark (show both true and predicted)
    # col 2: capability vs benchmark (show both true and predicted)
    # col 3: capability vs logit benchmark (show both true and predicted)

    pt = point.copy()
    augment_train_test_slaw(pt.slaw, pt.model, pt.split_train, pt.split_test)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5), squeeze=False)  # 4 columns
    ax_arr = ax[0]
    # plot in flop x-space and benchmark y-space
    plot_train_test(
        ax_arr[0],
        pt.split_train,
        pt.split_test,
        "log10 FLOP_opt",
        [
            Spe(f"{pt.slaw.benchmark}", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} pred", "Prediction", "C1"),
        ],
        y_label=pt.slaw.benchmark,
    )

    plot_train_test(
        ax_arr[1],
        pt.split_train,
        pt.split_test,
        "log10 FLOP_opt",
        [
            Spe(f"{pt.slaw.benchmark} logit", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} logit pred", "Prediction", "C1"),
        ],
        y_label=f"{pt.slaw.benchmark} logit",
    )

    plot_train_test(
        ax_arr[2],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{pt.slaw.benchmark}", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} pred", "Prediction", "C1"),
        ],
        y_label=pt.slaw.benchmark,
    )

    plot_train_test(
        ax_arr[3],
        pt.split_train,
        pt.split_test,
        "PC-1",
        [
            Spe(f"{pt.slaw.benchmark} logit", "Ground Truth", "C0"),
            Spe(f"{pt.slaw.benchmark} logit pred", "Prediction", "C1"),
        ],
        y_label=f"{pt.slaw.benchmark} logit",
    )

    plt.show()


def plot_linear_scaling_law(lin_data_point: BacktestDataPoint[LinearPC1Predictor]):
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
        ax_arr = ax[bench_idx]
        benchmark = pt.model.benchmarks[bench_idx]
        plot_train_test(
            ax_arr[0],
            pt.split_train,
            pt.split_test,
            "log10 FLOP_opt",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )
        plot_train_test(
            ax_arr[1],
            pt.split_train,
            pt.split_test,
            "PC-1 (linear)",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )

    plot_slaw(lin_data_point)


def plot_logit_scaling_law(logit_data_point: BacktestDataPoint[LogitPC1Predictor]):
    fig, ax = plt.subplots(
        len(logit_data_point.model.benchmarks),
        4,
        figsize=(4 * 4, len(logit_data_point.model.benchmarks) * 4),
        squeeze=False,
    )  # 1 columns

    for bench_idx, benchmark in enumerate(logit_data_point.model.benchmarks):
        ax_arr = ax[bench_idx]
        pt = logit_data_point.copy()
        augment_train_test_logit(pt.model, pt.split_train, pt.split_test)
        benchmark = pt.model.benchmarks[bench_idx]
        plot_train_test(
            ax_arr[0],
            pt.split_train,
            pt.split_test,
            "log10 FLOP_opt",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )

        plot_train_test(
            ax_arr[1],
            pt.split_train,
            pt.split_test,
            "log10 FLOP_opt",
            [
                Spe(f"{benchmark} logit", "Ground Truth", "C0"),
                Spe(f"{benchmark} logit pred", "Prediction", "C1"),
            ],
            y_label=f"{benchmark} logit",
        )

        plot_train_test(
            ax_arr[2],
            pt.split_train,
            pt.split_test,
            "PC-1 (logit)",
            [
                Spe(f"{benchmark}", "Ground Truth", "C0"),
                Spe(f"{benchmark} pred", "Prediction", "C1"),
            ],
            y_label=benchmark,
        )

        plot_train_test(
            ax_arr[3],
            pt.split_train,
            pt.split_test,
            "PC-1 (logit)",
            [
                Spe(f"{benchmark} logit", "Ground Truth", "C0"),
                Spe(f"{benchmark} logit pred", "Prediction", "C1"),
            ],
            y_label=f"{benchmark} logit",
        )

    plt.tight_layout()

    plt.show()

    plot_slaw(logit_data_point)


def plot_flop_scaling_law(flop_data_point: BacktestDataPoint[DirectLogFlopPredictor]):
    # fig, ax = plt.subplots(
    #     len(algprog_flop_data_point.model.benchmarks),
    #     4,
    #     figsize=(4 * 4, len(algprog_flop_data_point.model.benchmarks) * 4),
    #     squeeze=False,
    # )  # 1 columns

    # for bench_idx, benchmark in enumerate(logit_data_point.model.benchmarks):
    #     pt = logit_data_point.copy()
    #     augment_train_test_logit(pt.model, pt.split_train, pt.split_test)
    #     plot_logit_model(
    #         ax[bench_idx], bench_idx, pt.split_train, pt.split_test, pt.model
    #     )

    # plt.tight_layout()
    # plt.show()

    slaw = flop_data_point.slaw
    print("slaw beta", slaw.beta.item())
    print("slaw alpha", slaw.alpha.item())

    plt.plot(np.log10(slaw.train_losses[500:]), label="slaw")
    plt.show()

    plot_slaw(flop_data_point)


def plot_algprog_flop_scaling_law(
    algprog_flop_data_point: BacktestDataPoint[AlgprogLogFlopPredictor],
):
    # fig, ax = plt.subplots(
    #     len(algprog_flop_data_point.model.benchmarks),
    #     4,
    #     figsize=(4 * 4, len(algprog_flop_data_point.model.benchmarks) * 4),
    #     squeeze=False,
    # )  # 1 columns

    # for bench_idx, benchmark in enumerate(logit_data_point.model.benchmarks):
    #     pt = logit_data_point.copy()
    #     augment_train_test_logit(pt.model, pt.split_train, pt.split_test)
    #     plot_logit_model(
    #         ax[bench_idx], bench_idx, pt.split_train, pt.split_test, pt.model
    #     )

    # plt.tight_layout()
    # plt.show()

    model = algprog_flop_data_point.model
    print(f"PC1 = {model.m_c.item()} * log10(FLOPs) + {model.b_c.item()}")

    slaw = algprog_flop_data_point.slaw
    print("slaw beta", slaw.beta.item())
    print("slaw alpha", slaw.alpha.item())

    plt.plot(np.log10(slaw.train_losses[500:]), label="slaw")
    plt.show()

    plot_slaw(algprog_flop_data_point)


def plot_all_algprog_flop_fits(
    algprog_flop_data: BacktestData,
):
    n_split, n_bench = algprog_flop_data.results.shape
    fig, ax = plt.subplots(
        n_split * 3,
        n_bench,
        figsize=(4 * n_bench, 4 * 3 * n_split),
        squeeze=False,
    )
    for split_idx in range(n_split):
        for bench_idx in range(n_bench):
            bdp: BacktestDataPoint[AlgprogLogFlopPredictor] = algprog_flop_data.results[
                split_idx, bench_idx
            ]
            bdp = bdp.copy()
            augment_train_test_slaw(
                bdp.slaw, bdp.model, bdp.split_train, bdp.split_test
            )
            m_c = bdp.model.m_c.item()
            b_c = bdp.model.b_c.item()
            m_p = bdp.model.m_p.item()
            b_p = bdp.model.b_p.item()
            plot_train_test(
                ax[split_idx * 3 + 0, bench_idx],
                bdp.split_train,
                bdp.split_test,
                "log10 FLOP_opt",
                [
                    Spe(bdp.slaw.benchmark, "Ground Truth", "black"),
                    Spe(
                        f"{bdp.slaw.benchmark} pred",
                        f"S={m_c:.2f}C+{b_c:.2f}+{m_p:.2f}D+{b_p:.2f}",
                        "red",
                    ),
                ],
                y_label=bdp.slaw.benchmark,
            )
            ax[split_idx * 3 + 1, bench_idx].scatter(
                bdp.model.release_dates, bdp.model.slopes, label="Slopes"
            )
            ax[split_idx * 3 + 1, bench_idx].legend()
            ax[split_idx * 3 + 2, bench_idx].scatter(
                bdp.model.release_dates, bdp.model.y_intercepts, label="Intercepts"
            )
            xspace = np.linspace(
                min(bdp.model.release_dates, default=-1),
                max(bdp.model.release_dates, default=1),
                100,
            )
            ax[split_idx * 3 + 2, bench_idx].plot(
                xspace, m_p * xspace + b_p, label="Progress Fit"
            )
            ax[split_idx * 3 + 2, bench_idx].legend()
    fig.tight_layout()
    plt.show()


# %%

#####################################
# Train and fit global linear and logit models of PC-1
# Expanding Window
#####################################

ewbs = ExpandingWindowBacktestSplitter(
    min_train_size=10,
    test_size=10,
    increment=10,
    key="log10 FLOP_opt",
)


# %%

ewbs_lin_data = backtest_models(
    ewbs, LinearPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_lin_train_err, ewbs_lin_test_err = compute_test_train_error(ewbs_lin_data.results)

# %%
ewbs_logit_data = backtest_models(
    ewbs, LogitPC1Predictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_logit_train_err, ewbs_logit_test_err = compute_test_train_error(
    ewbs_logit_data.results
)


# %%
ewbs_flop_data = backtest_models(
    ewbs, DirectLogFlopPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_flop_train_err, ewbs_flop_test_err = compute_test_train_error(
    ewbs_flop_data.results
)


# %%
ewbs_algprog_flop_data = backtest_models(
    ewbs, LogFlopDatePredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_algprog_flop_train_err, ewbs_algprog_flop_test_err = compute_test_train_error(
    ewbs_algprog_flop_data.results
)

# %%
ewbs_elo_data = backtest_models(
    ewbs, DirectEloPredictor, openllm_elo_merged, openllm_elo_benchmarks
)
ewbs_elo_train_err, ewbs_elo_test_err = compute_test_train_error(ewbs_elo_data.results)


# %%

# # %%
# ewbs_lin_data = backtest_models(
#     ewbs, LinearPC1Predictor, base_llm_benchmark_eval, base_llm_benchmarks
# )
# ewbs_lin_train_err, ewbs_lin_test_err = compute_test_train_error(ewbs_lin_data.results)

# # %%
# ewbs_logit_data = backtest_models(
#     ewbs, LogitPC1Predictor, base_llm_benchmark_eval, base_llm_benchmarks
# )
# ewbs_logit_train_err, ewbs_logit_test_err = compute_test_train_error(
#     ewbs_logit_data.results
# )

# # %%
# ewbs_flop_data = backtest_models(
#     ewbs, DirectLogFlopPredictor, base_llm_benchmark_eval, base_llm_benchmarks
# )
# ewbs_flop_train_err, ewbs_flop_test_err = compute_test_train_error(
#     ewbs_flop_data.results
# )

# # %%
# ewbs_algprog_flop_data = backtest_models(
#     ewbs, AlgprogLogFlopPredictor, base_llm_benchmark_eval, base_llm_benchmarks
# )
# ewbs_algprog_flop_train_err, ewbs_algprog_flop_test_err = compute_test_train_error(
#     ewbs_algprog_flop_data.results
# )


# %%

######
# Compute the mean error of the linear and logit models
######

plot_errmatrix_comparison([ewbs_lin_data, ewbs_logit_data])

# %%

plot_comparison([ewbs_lin_data, ewbs_logit_data])

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

######
# Compute the mean error of the linear and flop models
######

plot_comparison([ewbs_lin_data, ewbs_flop_data])

print(f"Linear Train Error: {ewbs_lin_train_err.mean()}")
print(f"Flop Train Error: {ewbs_flop_train_err.mean()}")
print(f"Linear Test Error: {ewbs_lin_test_err.mean()}")
print(f"Flop Test Error: {ewbs_flop_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_lin_train_err.mean() - ewbs_flop_train_err.mean()) / ewbs_lin_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_lin_test_err.mean() - ewbs_flop_test_err.mean()) / ewbs_lin_test_err.mean() * 100:.2f}%"
)


# %%

######
# Compute the mean error of the linear and algprog flop models
######

plot_comparison([ewbs_lin_data, ewbs_algprog_flop_data])


print(f"Linear Train Error: {ewbs_lin_train_err.mean()}")
print(f"Algprog Flop Train Error: {ewbs_algprog_flop_train_err.mean()}")
print(f"Linear Test Error: {ewbs_lin_test_err.mean()}")
print(f"Algprog Flop Test Error: {ewbs_algprog_flop_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_lin_train_err.mean() - ewbs_algprog_flop_train_err.mean()) / ewbs_lin_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_lin_test_err.mean() - ewbs_algprog_flop_test_err.mean()) / ewbs_lin_test_err.mean() * 100:.2f}%"
)

# %%
# Compare flop and algprog flop
plot_comparison([ewbs_flop_data, ewbs_algprog_flop_data])

print(f"Flop Train Error: {ewbs_flop_train_err.mean()}")
print(f"Algprog Flop Train Error: {ewbs_algprog_flop_train_err.mean()}")
print(f"Flop Test Error: {ewbs_flop_test_err.mean()}")
print(f"Algprog Flop Test Error: {ewbs_algprog_flop_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_flop_train_err.mean() - ewbs_algprog_flop_train_err.mean()) / ewbs_flop_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_flop_test_err.mean() - ewbs_algprog_flop_test_err.mean()) / ewbs_flop_test_err.mean() * 100:.2f}%"
)


# %%
# Compare flop and elo
plot_comparison([ewbs_flop_data, ewbs_elo_data])

print(f"Flop Train Error: {ewbs_flop_train_err.mean()}")
print(f"Elo Train Error: {ewbs_elo_train_err.mean()}")
print(f"Flop Test Error: {ewbs_flop_test_err.mean()}")
print(f"Elo Test Error: {ewbs_elo_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_flop_train_err.mean() - ewbs_elo_train_err.mean()) / ewbs_flop_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_flop_test_err.mean() - ewbs_elo_test_err.mean()) / ewbs_flop_test_err.mean() * 100:.2f}%"
)

# %%
# Compare lin and elo
plot_comparison([ewbs_lin_data, ewbs_elo_data])

print(f"Linear Train Error: {ewbs_lin_train_err.mean()}")
print(f"Elo Train Error: {ewbs_elo_train_err.mean()}")
print(f"Linear Test Error: {ewbs_lin_test_err.mean()}")
print(f"Elo Test Error: {ewbs_elo_test_err.mean()}")

print(
    f"Train Percentage Improvement: {(ewbs_lin_train_err.mean() - ewbs_elo_train_err.mean()) / ewbs_lin_train_err.mean() * 100:.2f}%"
)
print(
    f"Test Percentage Improvement: {(ewbs_lin_test_err.mean() - ewbs_elo_test_err.mean()) / ewbs_lin_test_err.mean() * 100:.2f}%"
)
# %%

split_idx = 0
bench_idx = 1
plot_linear_scaling_law(ewbs_lin_data.results[split_idx, bench_idx])
# %%

split_idx = 0
bench_idx = 1
plot_logit_scaling_law(ewbs_logit_data.results[split_idx, bench_idx])

# %%

split_idx = 0
bench_idx = 0
plot_flop_scaling_law(ewbs_flop_data.results[split_idx, bench_idx])

# %%

split_idx = 2
bench_idx = 0
plot_algprog_flop_scaling_law(ewbs_algprog_flop_data.results[split_idx, bench_idx])


# %%
#####################################
# Train and fit family-specific linear models of PC-1
#####################################

plot_all_loss_curves(ewbs_logit_data)


# %%

plot_all_algprog_flop_fits(ewbs_algprog_flop_data)


# %%


# print ALL of the average errors:
# Linear, Logit, Algprog, Flop, Elo

print(
    f"Benchmark -> Linear PC1 -> Downstream Train MSE: {ewbs_lin_train_err.mean():.3f}"
)
print(f"Benchmark -> Linear PC1 -> Downstream Test MSE: {ewbs_lin_test_err.mean():.3f}")
print(
    f"Benchmark -> Logit PC1 -> Downstream Train MSE: {ewbs_logit_train_err.mean():.3f}"
)
print(
    f"Benchmark -> Logit PC1 -> Downstream Test MSE: {ewbs_logit_test_err.mean():.3f}"
)
print(f"(Flop, Date) -> Downstream Train MSE: {ewbs_algprog_flop_train_err.mean():.3f}")
print(f"(Flop, Date) -> Downstream Test MSE: {ewbs_algprog_flop_test_err.mean():.3f}")
print(f"Flop -> Downstream Train MSE: {ewbs_flop_train_err.mean():.3f}")
print(f"Flop -> Downstream Test MSE: {ewbs_flop_test_err.mean():.3f}")
print(f"Elo -> Downstream Train MSE: {ewbs_elo_train_err.mean():.3f}")
print(f"Elo -> Downstream Test MSE: {ewbs_elo_test_err.mean():.3f}")

print()
print()

print(
    f"Benchmark -> Linear PC1 -> Downstream Train RMSE: {ewbs_lin_train_err.mean()**0.5:.3f}"
)
print(
    f"Benchmark -> Linear PC1 -> Downstream Test RMSE: {ewbs_lin_test_err.mean()**0.5:.3f}"
)
print(
    f"Benchmark -> Logit PC1 -> Downstream Train RMSE: {ewbs_logit_train_err.mean()**0.5:.3f}"
)
print(
    f"Benchmark -> Logit PC1 -> Downstream Test RMSE: {ewbs_logit_test_err.mean()**0.5:.3f}"
)
print(
    f"(Flop, Date) -> Downstream Train RMSE: {ewbs_algprog_flop_train_err.mean()**0.5:.3f}"
)
print(
    f"(Flop, Date) -> Downstream Test RMSE: {ewbs_algprog_flop_test_err.mean()**0.5:.3f}"
)
print(f"Flop -> Downstream Train RMSE: {ewbs_flop_train_err.mean()**0.5:.3f}")
print(f"Flop -> Downstream Test RMSE: {ewbs_flop_test_err.mean()**0.5:.3f}")
print(f"Elo -> Downstream Train RMSE: {ewbs_elo_train_err.mean()**0.5:.3f}")
print(f"Elo -> Downstream Test RMSE: {ewbs_elo_test_err.mean()**0.5:.3f}")


plot_comparison(
    [
        ewbs_lin_data,
        ewbs_logit_data,
        ewbs_algprog_flop_data,
        ewbs_flop_data,
        ewbs_elo_data,
    ],
    expand=True,
)

# %%
plot_errmatrix_comparison(
    [
        ewbs_lin_data,
        ewbs_logit_data,
        ewbs_algprog_flop_data,
        ewbs_flop_data,
        ewbs_elo_data,
    ]
)
