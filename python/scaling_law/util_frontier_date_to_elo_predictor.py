from typing import override
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from util_frontier import Frontier, get_running_top_n
from util_obs_scaling_law_predictor import ScalingLaw


class FrontierDateToEloPredictor(Frontier):
    """
    This class directly passes through log FLOP
    """

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "Elo"]

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

        # train linear fit between date and elo on top 3
        frontier_date_vs_elo_df = get_running_top_n(
            train_df,
            "release_date",
            "Elo",
            3,
            "model",
        )
        
        m, b = np.polyfit(frontier_date_vs_elo_df["release_date"], frontier_date_vs_elo_df["Elo"], 1)
        self.date2elo_m = m
        self.date2elo_b = b
        
        # # train sigmoid fit between elo and downstream on top 3
        # frontier_elo_vs_downstream_df = get_running_top_n(
        #     train_df,
        #     "Elo",
        #     target_benchmark,
        #     3,
        #     "model",
        # )
        frontier_elo_vs_downstream_df = train_df
        
        train_model_scores = torch.tensor(frontier_elo_vs_downstream_df[benchmarks].values, dtype=torch.float32)
        
        self.target_benchmark = target_benchmark
        self.target_benchmark_floor = target_benchmark_floor
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors
        # we only store the logflop scores because they're the only thing we use
        # in M
        self.flop = nn.Buffer(train_model_scores[:, 0])

        self.slaw = ScalingLaw(
            benchmark=target_benchmark,
            floor=target_benchmark_floor,
            capability_scores=torch.tensor(frontier_elo_vs_downstream_df["Elo"].values, dtype=torch.float32),
            benchmark_scores=torch.tensor(frontier_elo_vs_downstream_df[target_benchmark].values, dtype=torch.float32),
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
        return self.slaw(test_scores[:, 0] * self.date2elo_m + self.date2elo_b)


    @override
    def predict_frontier_capability_scores(self, test_scores: torch.Tensor) -> torch.Tensor:
        return test_scores[:, 0] * self.date2elo_m + self.date2elo_b

    @override
    def capability_scores(self, train_df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(train_df["Elo"].values, dtype=torch.float32)

    @override
    def fit(self):
        self.slaw.fit()