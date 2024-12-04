import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

   
def get_running_top_n(
    df: pd.DataFrame, x_column: str, y_column: str, n: int, id_column: str
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(x_column)

    top_ids = set()
    x_values = df[x_column].unique()

    for x in x_values:
        data_until_x = df[df[x_column] <= x]
        top_n_at_date = data_until_x.nlargest(n, y_column)[id_column]
        top_ids.update(top_n_at_date)

    return df[df[id_column].isin(top_ids)]



class Frontier(nn.Module):
    """
    Parent class for all frontier predictors.
    """
    
    benchmarks: list[str]

    def __init__(
        self,
        benchmarks: list[str],
        benchmark_floors: list[float],
        target_benchmark: str,
        target_benchmark_floor: float,
        train_df: pd.DataFrame,
    ):
        super().__init__()
        pass

    @staticmethod
    def necessary_benchmarks() -> list[str]:
        """
        Return the list of benchmarks that are necessary for this predictor.
        These benchmarks must appear first in the tensor that is passed into the constructor,
        and the order of the benchmarks must be the same as the order of the benchmarks in this list.
        Only one of necessary_benchmarks and fixed_benchmarks should be implemented.
        """
        return []

    def predict_benchmark_scores(
        self,
        test_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict benchmark scores from capability scores.
        """
        raise NotImplementedError

    def fit(self):
        pass
