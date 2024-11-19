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

benchmarks, benchmark_floor = zip(*benchmark_data)

model_scores = [list(base_llm_benchmark_eval[benchmark]) for benchmark in benchmarks]


PC1_EPS = 1e-4
class AdjObsScalingLawPredictor(nn.Module):
    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floor: list[float],
        model_scores: list[list[float]],
    ):
        super().__init__()
        B = len(benchmarks)
        self.benchmarks = benchmarks
        # in B
        self.register_buffer(
            "benchmark_floor", torch.tensor(benchmark_floor, dtype=torch.float32)
        )
        self.benchmark_ceil_raw = nn.Parameter(
            torch.tensor([1.0] * B, dtype=torch.float32)
        )

        # in M x B
        self.register_buffer(
            "model_scores", torch.tensor(model_scores, dtype=torch.float32).T
        )

        # in B
        self.benchmark_weights = nn.Parameter(torch.ones(B, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.zeros(B, dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones(B, dtype=torch.float32))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        min_ceil = torch.max(self.model_scores, dim=0).values + PC1_EPS
        return (1 - min_ceil) * torch.sigmoid(self.benchmark_ceil_raw) + min_ceil

    @property
    def logit_scores(self) -> torch.Tensor:
        score_norm = (self.model_scores - self.benchmark_floor) / (
            (self.benchmark_ceil - self.benchmark_floor)
        )
        score_norm_floored = torch.clamp(score_norm, PC1_EPS, 1 - PC1_EPS)
        return torch.log(score_norm_floored / (1 - score_norm_floored))

    @property
    def pca_benchmark_weights(self) -> torch.Tensor:
        # perform PCA and get the first component, return it
        # logit_scores: M x B
        # benchmark_weights: B x 1
        U, S, V = torch.pca_lowrank(self.logit_scores)
        return -V[:, 0]

    def predict_capability_scores(self, logit_scores: torch.Tensor) -> torch.Tensor:
        # logit_scores: M x B
        # benchmark_weights: B x 1
        # benchmark_weights = self.get_benchmark_weights(logit_scores).unsqueeze(1)
        benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # M x 1
        capability_score = logit_scores @ benchmark_weights
        return capability_score.squeeze(1)

    def predict_benchmark_logit_scores(
        self, capability_scores: torch.Tensor
    ) -> torch.Tensor:
        # capability_scores: M x 1
        capability_scores = capability_scores.unsqueeze(1)
        # beta = 1 x B
        beta = self.beta.unsqueeze(0)
        # M x B
        predicted_logit_scores = capability_scores @ beta + self.alpha
        return predicted_logit_scores

    def predict_benchmark_scores(
        self, benchmark_logit_scores: torch.Tensor
    ) -> torch.Tensor:
        # in M x B
        return (self.benchmark_ceil - self.benchmark_floor) * torch.sigmoid(
            benchmark_logit_scores
        ) + self.benchmark_floor

    # loss due to the points that fall below the floor
    def intrinsic_loss(self) -> torch.Tensor:
        model_scores_floored = torch.max(self.model_scores, self.benchmark_floor)
        return F.mse_loss(self.model_scores, model_scores_floored)

    # compute loss
    def forward(self) -> torch.Tensor:
        logit_scores = self.logit_scores
        capability_scores = self.predict_capability_scores(logit_scores)
        benchmark_logit_scores = self.predict_benchmark_logit_scores(capability_scores)
        pred_benchmark_scores = self.predict_benchmark_scores(benchmark_logit_scores)
        return F.mse_loss(self.model_scores, pred_benchmark_scores)



class ObsScalingLawPredictor(nn.Module):
    def __init__(
        self,
        benchmarks: list[str],
        model_scores: list[list[float]],
    ):
        super().__init__()
        B = len(benchmarks)
        self.benchmarks = benchmarks

        # in M x B
        self.register_buffer(
            "model_scores", torch.tensor(model_scores, dtype=torch.float32).T
        )

        # in B
        self.benchmark_weights = nn.Parameter(torch.ones(B, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.zeros(B, dtype=torch.float32))
        self.beta = nn.Parameter(torch.ones(B, dtype=torch.float32))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        min_ceil = torch.max(self.model_scores, dim=0).values + PC1_EPS
        return (1 - min_ceil) * torch.sigmoid(self.benchmark_ceil_raw) + min_ceil

    @property
    def pca_benchmark_weights(self) -> torch.Tensor:
        # perform PCA and get the first component, return it
        # logit_scores: M x B
        # benchmark_weights: B x 1
        U, S, V = torch.pca_lowrank(self.model_scores)
        # gamma in B x 1
        return -V[:, 0]

    def predict_capability_scores(self) -> torch.Tensor:
        # benchmark_weights: B x 1
        benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # benchmark_weights = self.benchmark_weights.unsqueeze(1)
        # M x 1
        capability_score = self.model_scores @ benchmark_weights
        return capability_score.squeeze(1)

    def predict_benchmark_scores(self, capability_scores: torch.Tensor) -> torch.Tensor:
        # capability_scores: M x 1
        capability_scores = capability_scores.unsqueeze(1)
        # beta = 1 x B
        beta = self.beta.unsqueeze(0)
        # M x B
        predicted_logit_scores = capability_scores @ beta + self.alpha
        return predicted_logit_scores

    # compute loss
    def forward(self) -> torch.Tensor:
        capability_scores = self.predict_capability_scores()
        pred_benchmark_scores = self.predict_benchmark_scores(capability_scores)
        return F.mse_loss(self.model_scores, pred_benchmark_scores)


# %%

# Train AdjObs model and then create plot