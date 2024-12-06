from typing import override
import pandas as pd
import torch
import torch.nn as nn

from util_frontier import Frontier, get_running_top_n
from util_obs_scaling_law_predictor import ScalingLaw


class FrontierDatePredictor(Frontier):
    """
    This class directly passes through log FLOP
    """

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date"]

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        target_benchmark: str,
        target_benchmark_floor: float,
        train_df: pd.DataFrame,
    ):  
        super().__init__(
            benchmarks,
            benchmark_floors,
            target_benchmark,
            target_benchmark_floor,
            train_df,
        )

        assert benchmarks[:len(self.necessary_benchmarks())] == self.necessary_benchmarks()

        # train on top 3
        frontier_train_df = get_running_top_n(
            train_df,
            "release_date",
            target_benchmark,
            3,
            "model",
        )
        
        train_model_scores = torch.tensor(frontier_train_df[benchmarks].values, dtype=torch.float32)
        
        self.target_benchmark = target_benchmark
        self.target_benchmark_floor = target_benchmark_floor
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors
        # we only store the logflop scores because they're the only thing we use
        # in M
        self.release_dates = nn.Buffer(train_model_scores[:, 0])

        self.slaw = ScalingLaw(
            benchmark=target_benchmark,
            floor=target_benchmark_floor,
            capability_scores=torch.tensor(frontier_train_df["release_date"].values, dtype=torch.float32),
            benchmark_scores=torch.tensor(frontier_train_df[target_benchmark].values, dtype=torch.float32),
        )

        self.train_losses = []

    @override
    def predict_benchmark_scores(
        self,
        test_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The only benchmark is the release_date, which is the capability score.
        """
        return self.slaw(test_scores[:, 0])

    @override
    def fit(self):
        self.slaw.fit()