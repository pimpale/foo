import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ObsScalingLawPredictor(nn.Module):
    """
    Parent class for all observational scaling law predictors.
    """

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

PC1_EPS = 1e-4
class ScalingLaw(nn.Module):
    def __init__(
        self,
        floor: float,
        capability_scores: torch.Tensor,
        benchmark_scores: torch.Tensor,
    ):
        super().__init__()
        self.register_buffer(
            "benchmark_floor", torch.tensor(floor, dtype=torch.float32)
        )
        self.register_buffer("capability_scores", capability_scores)
        self.register_buffer("benchmark_scores", benchmark_scores)
        self.benchmark_ceil_raw = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    @property
    def benchmark_ceil(self) -> torch.Tensor:
        min_ceil = torch.clamp(torch.max(self.benchmark_scores, dim=0).values, 0.8, 1)
        return (1 - min_ceil) * torch.sigmoid(self.benchmark_ceil_raw) + min_ceil

    def predict_benchmark_logit_scores(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * x + self.alpha

    def forward(self, x):
        return (
            self.benchmark_ceil - self.benchmark_floor
        ) * self.predict_benchmark_logit_scores(x) + self.benchmark_floor

    @torch.compile(fullgraph=True)
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
        optimizer = optim.Adam(params=self.parameters(), lr=5e-2, fused=True)
        for i in range(epochs):
            optimizer.zero_grad()
            l = self.train_loss()
            l.backward()
            optimizer.step()
        self.eval()
