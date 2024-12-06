from typing import override
import numpy as np
import torch
import torch.nn as nn
from util_logit_obs_scaling_law_predictor import LogitPC1Predictor
from util_obs_scaling_law_predictor import ObsScalingLawPredictor


class LogFlopDatePredictor(ObsScalingLawPredictor):
    """
    This class tries to predict PC1 capability scores using a linear approximation with both release date and amount of compute used.
    """

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return [
            "release_date",
            "log10 FLOP_opt",
        ]

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        train_model_scores: torch.Tensor,
    ):
        super().__init__(
            benchmarks,
            benchmark_floors,
            train_model_scores,
        )

        # check that required benchmarks are there
        assert (
            benchmarks[: len(self.necessary_benchmarks())]
            == self.necessary_benchmarks()
        )
        # we need at least one other benchmark
        assert len(benchmarks) > 3
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors

        # initialize logit predictor
        self.logit_pc1_predictor = LogitPC1Predictor(
            benchmarks=benchmarks[3:],
            benchmark_floors=benchmark_floors[3:],
            train_model_scores=train_model_scores[:, 3:],
        )

        # release dates of each model
        # in M
        self.D = nn.Buffer(train_model_scores[:, 0])

        # in M
        self.C = nn.Buffer(train_model_scores[:, 1])

        # linear approximation constants
        self.m_p = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.b_p = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.m_c = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.b_c = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        self.train_losses = []
        self.release_dates = []
        self.slopes = []
        self.y_intercepts = []

    @override
    def predict_capability_scores_from_model_scores(
        self,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The capability score is just the logflops score.
        """
        D_test = model_scores[:, 0]
        C_test = model_scores[:, 1]
        # compute a linear approximation of S_p considering both
        # the algorithmic progress since the epoch date
        # and the amount of compute used
        return (self.m_c * C_test + self.b_c) + (self.m_p * D_test + self.b_p)

    @override
    def fit(self):
        self.logit_pc1_predictor.fit()

        S_p = self.logit_pc1_predictor.predict_capability_scores_from_model_scores(
            self.logit_pc1_predictor.train_model_scores
        )

        # fit a linear regression going from C to S_noprog
        compute = self.C.detach().cpu().numpy()
        date = self.D.detach().cpu().numpy()
        capability_at_epoch = S_p.detach().cpu().numpy()
        m_c, m_p, b_c = np.linalg.lstsq(
            np.vstack([compute, date, np.ones(len(S_p))]).T,
            capability_at_epoch,
            rcond=None,
        )[0]
        self.m_c.data = torch.tensor(m_c, dtype=torch.float32)
        self.m_p.data = torch.tensor(m_p, dtype=torch.float32)
        self.b_c.data = torch.tensor(b_c, dtype=torch.float32)
