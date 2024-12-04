from typing import override
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from util_frontier import Frontier, get_running_top_n, get_running_top_n_2d
from util_obs_scaling_law_predictor import ScalingLaw


class FrontierFlopDatePredictor(Frontier):
    """
    This class directly passes through log FLOP and Date
    """

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["log10 FLOP_opt", "release_date"]

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

        assert benchmarks == self.fixed_benchmarks()

        # train on top 3
        frontier_train_df = get_running_top_n_2d(
            train_df,
            "log10 FLOP_opt", 
            "release_date",
            target_benchmark,
            3,
            "model",
        )
        
        # fit a 2d linear scaling law to optimally convert FLOP and Date to a combined metric
        m1, m2, b = np.linalg.lstsq(
            frontier_train_df[["log10 FLOP_opt", "release_date"]].values.T,
            frontier_train_df[target_benchmark].to_numpy(),
        )[0]
        self.m_compute = m1
        self.m_date = m2
        self.b = b

        # add the combined metric to the df
        train_df["combined_date_compute"] = m1 * train_df["log10 FLOP_opt"] + m2 * train_df["release_date"] + b        
        
        self.target_benchmark = target_benchmark
        self.target_benchmark_floor = target_benchmark_floor
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors

        self.slaw = ScalingLaw(
            benchmark=target_benchmark,
            floor=target_benchmark_floor,
            capability_scores=torch.tensor(train_df["combined_date_compute"].values, dtype=torch.float32),
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
        return self.slaw(test_scores[:, 0] * self.m_compute + test_scores[:, 1] * self.m_date + self.b)

    @override
    def fit(self):
        self.slaw.fit()