from typing import override
import numpy as np
import torch
import torch.nn as nn
from util_logit_obs_scaling_law_predictor import LogitPC1Predictor
from util_obs_scaling_law_predictor import ObsScalingLawPredictor


class AlgprogLogFlopPredictor(ObsScalingLawPredictor):
    """
    This class tries to predict PC1 capability scores using a linear approximation with both release date and amount of compute used.
    """

    @override
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["release_date", "log10 FLOPs_opt_Besiroglu (1E21)", "family_idx"]

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
        assert benchmarks[:3] == [
            "release_date",
            "log10 FLOPs_opt_Besiroglu (1E21)",
            "family_idx",
        ]
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

        # family indexes of each model
        # in M
        self.F = nn.Buffer(train_model_scores[:, 2])

        # linear approximation constants
        self.m_p = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.b_p = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.m_c = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.b_c = nn.Parameter(torch.tensor(0, dtype=torch.float32))

        self.train_losses = []

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



    # @override
    # def fit(self):
    #     self.logit_pc1_predictor.fit()

    #     S_p = self.logit_pc1_predictor.predict_capability_scores_from_model_scores(
    #         self.logit_pc1_predictor.train_model_scores
    #     )
        
    #     # fit a linear regression going from C to S_noprog
    #     compute = self.C.detach().cpu().numpy()
    #     capability_at_epoch = S_p.detach().cpu().numpy()
    #     m_c, b_c = np.polyfit(compute, capability_at_epoch, 1)
    #     self.m_c.data = torch.tensor(m_c, dtype=torch.float32)
    #     self.b_c.data = torch.tensor(b_c, dtype=torch.float32)

    @override
    def fit(self):
        self.logit_pc1_predictor.fit()

        S_p = self.logit_pc1_predictor.predict_capability_scores_from_model_scores(
            self.logit_pc1_predictor.train_model_scores
        )

        # not vectorized because there are only a few families
        release_date_list = []
        flops_opt_vs_pc1_b_list = []
        for family in torch.unique(self.F):
            family_mask = self.F == family
            
            # discard families with only one model
            print(family)
            if family_mask.sum() <= 1:
                continue
            
            family_capability_scores = S_p[family_mask].detach().cpu().numpy()
            family_compute_scores = self.C[family_mask].detach().cpu().numpy()
            family_release_date = self.D[family_mask].detach().mean().item()

            # fit linear regression
            flops_opt_vs_pc1_m, flops_opt_vs_pc1_b = np.polyfit(
                family_compute_scores, family_capability_scores, 1
            )
            # store the y-intercept (which is the algorithmic progress offset)
            # we can think of it as an increase in performance from some epoch date
            flops_opt_vs_pc1_b_list.append(flops_opt_vs_pc1_b)
            # store the release date
            release_date_list.append(family_release_date)
        
        # fit a linear regression going from date to the algorithmic progress constant
        m_p, b_p = np.polyfit(release_date_list, flops_opt_vs_pc1_b_list, 1)
        self.m_p.data = torch.tensor(m_p, dtype=torch.float32)
        self.b_p.data = torch.tensor(b_p, dtype=torch.float32)
        
        # compute S_p at the epoch date 
        S_noprog = S_p - (self.m_p * self.D + self.b_p)
        
        # fit a linear regression going from C to S_noprog
        compute = self.C.detach().cpu().numpy()
        capability_at_epoch = S_noprog.detach().cpu().numpy()
        m_c, b_c = np.polyfit(compute, capability_at_epoch, 1)
        self.m_c.data = torch.tensor(m_c, dtype=torch.float32)
        self.b_c.data = torch.tensor(b_c, dtype=torch.float32)
