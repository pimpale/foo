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


class ScalingLaw(nn.Module):
    def __init__(self, floor: float):
        super().__init__()
        self.register_buffer("floor", torch.tensor(floor, dtype=torch.float32))
        self.h = nn.Parameter(torch.Tensor(1))
        self.alpha = nn.Parameter(torch.Tensor(0))
        self.beta = nn.Parameter(torch.Tensor(1))

    def forward(self, x):
        return self.h * torch.sigmoid(self.beta * x + self.alpha) + self.floor

    def fit(
        self,
        # shape: M
        capability_scores: torch.Tensor,
        # shape: M
        benchmark_scores: torch.Tensor,
        # how many epochs to train for
        epochs: int = 5000,
    ):
        """
        Fit the scaling law to the provided model and benchmark scores.
        """
        optimizer = optim.Adam(params=self.parameters(), lr=1e-2)
        for i in range(epochs):
            optimizer.zero_grad()
            l = F.mse_loss(
                self.forward(capability_scores),
                benchmark_scores,
            )
            l.backward()
            optimizer.step()
            if i % 500 == 0:
                print(l.item())
        self.eval()
