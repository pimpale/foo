import torch
import torch.nn as nn
from util_obs_scaling_law_predictor import ObsScalingLawPredictor

class DirectEloPredictor(ObsScalingLawPredictor):
    """
    This class directly passes through Chatbot Arena Elo
    """
    
    @staticmethod
    def necessary_benchmarks() -> list[str]:
        return ["Elo"]
    
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
        
        assert benchmarks == ["Elo"]
        
        self.benchmarks = benchmarks
        self.benchmark_floors = benchmark_floors
        # we only store the elo scores because they're the only thing we use
        # in M
        self.elo_scores = nn.Buffer(train_model_scores[:, 0])
        
        self.train_losses = []
    
    def predict_capability_scores_from_model_scores(
        self,
        model_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The capability score is just the Elo score.
        """
        return model_scores[:, 0]
    
    def predict_benchmark_scores_from_capability_scores(
        self,
        capability_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        The only benchmark is the Elo, which is the capability score.
        """
        return capability_scores