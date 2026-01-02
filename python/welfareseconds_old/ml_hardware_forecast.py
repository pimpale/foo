#%%

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import math
from datetime import datetime
import numpy as np


def compute_rank(df, column, out_column = 'rank'):
    """
    For each model in time order, rank its cost_per_hour among all previously seen models.
    Rank 1 = highest cost_per_hour at release.
    """
    ranks = []
    seen_costs = []
    for cost in df[column]:
        if not np.isfinite(cost):
            ranks.append(np.nan)
            continue
        rank = 1 + sum(c >= cost for c in seen_costs)
        ranks.append(rank)

        seen_costs.append(cost)
        # count how many seen so far have cost >= current
    df = df.copy()
    df[out_column] = ranks
    return df

def find_trend(df, y_col, n_boot=100_000, seed = None):
    rng = np.random.default_rng(seed)
    df_filtered = df[df[y_col].notna()]

    x = df_filtered['year_float'].to_numpy()
    y = np.log(df_filtered[y_col]).to_numpy()
    n = x.size

    # bootstrap indices: shape (n_boot, n)
    idx = rng.integers(0, n, size=(n_boot, n))

    x_samp = x[idx]
    y_samp = y[idx]

    # closed-form OLS slope: β = (n Σxy − Σx Σy) / (n Σx² − (Σx)²)
    sum_x  = x_samp.sum(axis=1)
    sum_y  = y_samp.sum(axis=1)
    sum_xy = (x_samp * y_samp).sum(axis=1)
    sum_x2 = (x_samp**2).sum(axis=1)

    # Add a small epsilon to the denominator to avoid division by zero
    denominator = n * sum_x2 - sum_x**2
    slopes = (n * sum_xy - sum_x * sum_y) / (denominator + 1e-30)

    # Filter out invalid slopes (NaN or Inf)
    slopes = slopes[np.isfinite(slopes)]

    # Check if there are enough valid slopes to calculate percentiles
    assert slopes.size > 0, "Not enough valid slopes to calculate percentiles."
    lo, mid, hi = np.exp(np.percentile(slopes, [5,50, 95]))
    return mid, (lo, hi)


def trendline_from_growth(df, y_col, yoy_growth, results_x = None):
  slope = np.log(yoy_growth)
  df_filtered = df[df[y_col].notna() & df['year_float'].notna()].copy()
  x = df_filtered['year_float']

  if results_x is None:
    results_x = x

  y = np.log(df_filtered[y_col])

  # Adjust y values based on the forced slope
  y_adjusted = y - slope * x
  # Fit OLS model with only an intercept on the adjusted y values
  model_forced = sm.OLS(y_adjusted, sm.add_constant(np.zeros_like(x))).fit()
  # Get the intercept of the forced model
  intercept_forced = model_forced.params[0]
  # Calculate the trend line values with the forced slope and fitted intercept

  trend_line_forced = np.exp(intercept_forced + slope * results_x)
  return results_x, trend_line_forced

def find_and_graph_trend(df, y_col, extended_df = None, name = None, seed = None, n_boot=100_000):
  if name is None:
    name = y_col
  if extended_df is None:
    extended_df = df
  mid, (lo, hi) = find_trend(df, y_col, n_boot, seed)
  #print(f"Trend for {name}: {mid:.1f}x ({lo:.1f}x-{hi:.1f}x)")
  plt.figure()

  # DOES THIS HANDLE FILTERING RIGHT??? I THINK SO
  plt.scatter(extended_df['year_float'], extended_df[y_col])

  X = extended_df['year_float']

  # Use the function to get the forced trend line
  x_low, trend_line_low = trendline_from_growth(df, y_col, lo, X)
  x_high, trend_line_high = trendline_from_growth(df, y_col, hi, X)
  x_mid, trend_line_mid = trendline_from_growth(df, y_col, mid, X)

  plt.plot(x_mid, trend_line_mid, color='red', label=f'Median ({mid:.1f}x)')
  plt.fill_between(x_low, trend_line_low, trend_line_high, color='red', alpha=0.2, label=f'90% CI ({lo:.1f}x-{hi:.1f}x)')


  plt.yscale('log')
  plt.xlabel('Year')
  plt.ylabel(y_col)
  plt.title(f'Trend of {name}: {mid:.1f}x ({lo:.1f}x-{hi:.1f}x)')
  plt.grid(True)
  plt.legend()
  plt.show()
  #return mid, (lo, hi)


#%%


width_map = {
    'FP64 (double precision) performance (FLOP/s)': 64, 'FP32 (single precision) performance (FLOP/s)': 32,
    'FP16 (half precision) performance (FLOP/s)': 16, 'TF32 (TensorFloat-32) performance (FLOP/s)': 19,
    'Tensor-FP16/BF16 performance (FLOP/s)': 16, 'INT16 performance (OP/s)': 16,
    'INT8 performance (OP/s)': 8, 'INT4 performance (OP/s)': 4,
}
def tpp(row):
    best=None
    for col,w in width_map.items():
        if col in row and pd.notna(row[col]):
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.notna(val):
                cur = w*val
                if best is None or cur > best:
                    best = cur
    return best


hw = pd.read_csv('./ml_hardware.csv', parse_dates=['Release date'])
hw['TPP'] = hw.apply(tpp, axis=1)
hw['Price'] = pd.to_numeric(hw['Release price (USD)'], errors='coerce')
hw['tpp_per_dollar'] = hw['TPP'] / hw['Price']
#hw['tensor per dollar'] = hw[['FP16 (half precision) performance (FLOP/s)',
#                               'Tensor-FP16/BF16 performance (FLOP/s)']].max(axis=1)/hw['Price']
hw['tpp_per_watt'] =  hw['TPP'] / hw['TDP (W)']


hw = hw[hw['Hardware name'].notna()] # Yeah idk why we have an item without a name but it seems to be fake
hw = hw[hw['Release date'].notna()]


hw = hw.sort_values('Release date').reset_index(drop=True)
hw['year_float'] = hw['Release date'].dt.year + (hw['Release date'].dt.dayofyear - 1)/365.25


# %%

hw = hw[~hw['Hardware name'].str.contains('NVIDIA.*(GeForce|Quadro|RTX|TITAN|Titan|GTX)', case=False)]
hw = hw[(~(hw['Hardware name'].str.startswith('AMD'))) |
        (hw['Hardware name'].str.contains('AMD.*(FirePro S|Instinct)', case=False))]

# hw = hw[hw['Price'].notna() | hw['TDP (W)'].notna()]

# %% 

# Hardware Results

stats_hw = ('tpp_per_dollar', 'tpp_per_watt')
for stat in stats_hw:
  mid, (lo, hi) = find_trend(hw, stat)
  print(f"Trend for {stat}: {mid:.1f}x ({lo:.1f}x-{hi:.1f}x)")
  print(f"Data points for {stat}: {hw[stat].count()}")
  print("")

find_and_graph_trend(hw, 'tpp_per_dollar')
# %%
