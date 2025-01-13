from typing import override
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from util_frontier import Frontier, get_running_top_n
from util_linear_obs_scaling_law_predictor import LinearPC1Predictor
from util_obs_scaling_law_predictor import ScalingLaw


class FrontierDateToPC1Predictor(Frontier):
    """
    This class first predicts date to pc1, and then predicts pc1 to downstream
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

        pc1_benchmarks = benchmarks[1:]
        pc1_benchmark_floors = benchmark_floors[1:]

        train_model_scores = torch.tensor(train_df[pc1_benchmarks].values, dtype=torch.float32)

        # fit linear PC1 predictor on all models on all benchmarks except for release_date
        self.linpc1 = LinearPC1Predictor(
            benchmarks=pc1_benchmarks,
            benchmark_floors=pc1_benchmark_floors,
            train_model_scores=train_model_scores,
        )
        self.linpc1.fit()

        # insert PC1 scores into the df
        train_df["PC1"] = self.linpc1.predict_capability_scores_from_model_scores(train_model_scores).detach().numpy()
        

        # train linear fit between date and pc1 on top 3
        frontier_date_vs_pc1_df = get_running_top_n(
            train_df,
            "release_date",
            "PC1",
            3,
            "model",
        )
        
        # fit linear model
        m, b = np.polyfit(frontier_date_vs_pc1_df["release_date"], frontier_date_vs_pc1_df["PC1"], 1)
        self.date2pc1_m = m
        self.date2pc1_b = b
        
        # train sigmoid fit between pc1 and downstream on top 3
        frontier_pc1_vs_downstream_df = get_running_top_n(
            train_df,
            "PC1",
            target_benchmark,
            3,
            "model",
        )
                
        self.target_benchmark = target_benchmark
        self.target_benchmark_floor = target_benchmark_floor
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors


        self.slaw = ScalingLaw(
            benchmark=target_benchmark,
            floor=target_benchmark_floor,
            capability_scores=torch.tensor(frontier_pc1_vs_downstream_df["PC1"].values, dtype=torch.float32),
            benchmark_scores=torch.tensor(frontier_pc1_vs_downstream_df[target_benchmark].values, dtype=torch.float32),
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
        return self.slaw(test_scores[:, 0] * self.date2pc1_m + self.date2pc1_b)


    @override
    def predict_frontier_capability_scores(self, test_scores: torch.Tensor) -> torch.Tensor:
        return test_scores[:, 0] * self.date2pc1_m + self.date2pc1_b

    @override
    def capability_scores(self, train_df: pd.DataFrame) -> torch.Tensor:
        return torch.tensor(train_df["Elo"].values, dtype=torch.float32)

    @override
    def fit(self):        
        self.slaw.fit()