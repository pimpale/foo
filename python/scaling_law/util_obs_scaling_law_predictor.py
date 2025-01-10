from copy import deepcopy
import time
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ObsScalingLawPredictor(nn.Module):
    """
    Parent class for all observational scaling law predictors.
    """

    benchmarks: list[str]

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
    ):
        super().__init__()
        pass

    
    @staticmethod
    def fixed_benchmarks() -> list[str] | None:
        """
        Return the list of benchmarks that are fixed for this predictor.
        These benchmarks must appear after the necessary benchmarks in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        """
        return None

    @staticmethod
    def necessary_benchmarks() -> list[str]:
        """
        Return the list of benchmarks that are necessary for this predictor.
        These benchmarks must appear first in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        Only one of necessary_benchmarks and fixed_benchmarks should be implemented.
        """
        return []


    def predict_capability_scores_from_model_scores(
        self,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict capability scores directly from model scores.
        """
        raise NotImplementedError

    def predict_benchmark_scores_from_capability_scores(
        self,
        capability_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict benchmark scores from capability scores.
        """
        raise NotImplementedError

    def fit(self):
        pass

PC1_EPS = 1e-4


class ScalingLaw(nn.Module):
    def __init__(
        self,
        benchmark: str,
        floor: float,
        capability_scores: torch.Tensor,
        benchmark_scores: torch.Tensor,
    ):
        super().__init__()
        self.train_losses = []
        self.benchmark = benchmark
        self.benchmark_floor = nn.Buffer(torch.tensor(floor, dtype=torch.float32))
        self.capability_scores = nn.Buffer(capability_scores)
        self.benchmark_scores = nn.Buffer(benchmark_scores)
        self.benchmark_ceil_raw = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.alpha = nn.Parameter(-torch.mean(capability_scores)/torch.std(capability_scores))
        self.beta = nn.Parameter(1/torch.std(capability_scores))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        min_ceil = torch.clamp(torch.max(self.benchmark_scores, dim=0).values, 0.8, 1)
        return (1 - min_ceil) * torch.sigmoid(self.benchmark_ceil_raw) + min_ceil

    def predict_logit_scores(self, x: torch.Tensor) -> torch.Tensor:
        score_norm = (x - self.benchmark_floor) / (
            (self.benchmark_ceil - self.benchmark_floor)
        )
        score_norm_floored = torch.clamp(score_norm, PC1_EPS, 1 - PC1_EPS)
        return torch.log(score_norm_floored / (1 - score_norm_floored))

    def predict_benchmark_logit_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * x + self.alpha

    def forward(self, x):
        return (self.benchmark_ceil - self.benchmark_floor) * torch.sigmoid(
            self.predict_benchmark_logit_scores(x)
        ) + self.benchmark_floor

    # @torch.compile(fullgraph=True, dynamic=True)
    def train_loss(self):
        return F.mse_loss(self(self.capability_scores), self.benchmark_scores)

    def fit(
        self,
        # how many epochs to train for
        epochs: int = 5000,
    ):
        """
        Fit the scaling law to the provided model and benchmark scores.
        """
        optimizer = optim.Adam(params=self.parameters(), lr=1e-2, fused=True)
        best_train_loss = float("inf")
        best_state_dict = self.state_dict()
        for i in range(epochs):
            optimizer.zero_grad()
            l = self.train_loss()
            if l < best_train_loss:
                best_train_loss = l
                best_state_dict = deepcopy(self.state_dict())
            l.backward()
            torch.nn.utils.clip_grad_value_(self.parameters(), 5)
            optimizer.step()
            self.train_losses.append(l.item())
        # load best state dict
        self.load_state_dict(best_state_dict)
        # get last loss
        self.train_losses.append(self.train_loss().item())
