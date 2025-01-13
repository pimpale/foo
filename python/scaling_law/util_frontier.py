import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from util_obs_scaling_law_predictor import ScalingLaw

   
def get_running_top_n(
    df: pd.DataFrame, x_column: str, y_column: str, n: int, id_column: str
) -> pd.DataFrame:
    """
    This function returns all models that are in the top n of y_column at any point in time, where time is given by x_column.
    """
    top_ids = set()
    x_values = df[x_column].unique()

    for x in x_values:
        data_until_x = df[df[x_column] <= x]
        top_n_at_date = data_until_x.nlargest(n, y_column)[id_column]
        top_ids.update(top_n_at_date)

    return df[df[id_column].isin(top_ids)]

def get_running_top_n_2d(
    df: pd.DataFrame, x_column: str, y_column: str, z_column: str, n: int, id_column: str
) -> pd.DataFrame:
    """
    This function returns all models that are in the top n of z_column for any x,y pair in xy_columns.
    """
    top_ids = set()
    xy_values = df[[x_column, y_column]].drop_duplicates().values

    for x, y in xy_values:
        data_until_xy = df[(df[x_column] <= x) & (df[y_column] <= y)]
        top_n_at_date = data_until_xy.nlargest(n, z_column)[id_column]
        top_ids.update(top_n_at_date)

    return df[df[id_column].isin(top_ids)]


def vectorized_highest_score(df, x_column: str, x_column_thresholds: np.ndarray, key: str):
    """
    Vectorized function to return the highest `key` score for each threshold.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    x_column (str): The column to threshold using x_column_thresholds.
    x_column_thresholds (np.ndarray): Array of thresholds.
    key (str): The key to search for the highest score.

    Returns:
    np.ndarray: Array of highest `key` scores.
    """
    # Create an array to store the highest scores
    highest_scores = np.zeros(len(x_column_thresholds))

    for i, x in enumerate(x_column_thresholds):
        mask = df[x_column] <= x
        if mask.any():
            highest_scores[i] = df.loc[mask, key].max()
        else:
            highest_scores[i] = np.nan  # or some other placeholder for no data

    return highest_scores

def vectorized_highest_score_2d(df, x_column, x_column_thresholds,  y_column, y_column_thresholds,  key):
    """
    Vectorized function to return the highest `key` score for each combination of x_column and y_column.

    Parameters:
    df (pd.DataFrame): The dataframe to search.
    x_column (str): The column to threshold using x_column_thresholds.
    x_column_thresholds (np.ndarray): Array of thresholds
    y_column (str): The column to threshold using y_column_thresholds.
    y_column_thresholds (np.ndarray): Array of thresholds.
    key (str): The key to search for the highest score.

    Returns:
    np.ndarray: 2D array of highest `key` scores.
    """
    # Create a 2D array to store the highest scores
    highest_scores = np.zeros((len(x_column_thresholds), len(y_column_thresholds)))

    for i, x_threshold in enumerate(x_column_thresholds):
        mask = (df[x_column] <= x_threshold)
        for j, y_threshold in enumerate(y_column_thresholds):
            combined_mask = mask & (df[y_column] <= y_threshold)
            if combined_mask.any():
                highest_scores[i, j] = df.loc[combined_mask, key].max()
            else:
                highest_scores[i, j] = np.nan  # or some other placeholder for no data

    return highest_scores

class Frontier(nn.Module):
    """
    Parent class for all frontier predictors.
    """
    
    benchmarks: list[str]
    slaw: ScalingLaw

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

    def predict_frontier_capability_scores(
        self,
        test_scores: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict benchmark scores from capability scores.
        """
        raise NotImplementedError

    def capability_scores(
        self,
        train_df: pd.DataFrame,
    ):
        """
        Calculate capability scores from the training data.
        """
        raise NotImplementedError


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
