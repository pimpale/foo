import torch
import torch.nn as nn
import torch.nn.functional as F

PC1_EPS = 1e-4
class LinearObsScalingLawPredictor(nn.Module):
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

    def fit(self, optimizer, epochs=1000):
        for i in range(5000):
            optimizer.zero_grad()
            l = self.forward()
            l.backward()
            optimizer.step()
            if i % 100 == 0:
                print(l.item())