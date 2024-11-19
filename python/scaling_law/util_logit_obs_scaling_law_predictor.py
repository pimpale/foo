import torch
import torch.nn as nn
import torch.nn.functional as F

PC1_EPS = 1e-4
class LogitObsScalingLawPredictor(nn.Module):
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

    def fit(self, optimizer):
        for i in range(5000):
            optimizer.zero_grad()
            l = self.forward()
            l.backward()
            optimizer.step()
            if i % 100 == 0:
                print(l.item())
