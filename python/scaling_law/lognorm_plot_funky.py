#%%

from typing import Literal
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact
import ipywidgets as widgets
import pandas as pd
from collections import defaultdict

def sigmoid(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - shift)))

def lognormal_cdf(x, loc, sigma, mu):
    return lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu))

def lognormal_pdf(x, loc, sigma, mu):
    return lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu))


def get_sigmoid_parameters(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        sigmoid,
        x_values,
        y_values,
        p0=[
            # slope
            1,
            # shift
            0,
        ],
        bounds=([0, -100], [10, 100]),
        maxfev=5000
    )
    return popt

def get_lognormal_cdf_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        lognormal_cdf,
        x_values,
        y_values,
        p0=[
            # center at the median
            -10, 0.4, 2.6
        ],
        bounds=([-100, 0.01, 0.1], [20, 10, 10]),
        maxfev=5000
    )
    return popt


#%%

results  = pd.read_csv("./results.csv")

pc1_scores = pd.read_csv('./leaderboard_pca_scores.csv')

# construct a dict mapping from model name to the scores
pc1_scores_dict = dict(zip(pc1_scores['model'], pc1_scores['PC1']))

# construct a dict of dicts 
results_dict = defaultdict(lambda: defaultdict(list))
for idx, row in results.iterrows():
    # print(row)
    task = row['taskFamily'] +' ' + row['taskName']
    if row['score'] != -1:
        results_dict[task][row['model']] += [row['score']]

# compute average success rate and variance for each task
results_dict_avg_std = {}
for task, model_scores in results_dict.items():
    avg_scores = {}
    for model, scores in model_scores.items():
        if model in pc1_scores_dict:
            avg_scores[model] = {
                'pc1': pc1_scores_dict[model],
                'avg': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
    results_dict_avg_std[task] = avg_scores    

# fit a sigmoid for each task:
params = []
for task, model_scores in results_dict_avg_std.items():
    
    # discard tasks which have all zeros or ones
    if all(scores['avg'] == 0 for model, scores in model_scores.items()):
        continue
    
    x_values = [scores['pc1'] for model, scores in model_scores.items()]
    y_values = [scores['avg'] for model, scores in model_scores.items()]
    fitted_params = get_sigmoid_parameters(x_values, y_values)
    params.append(fitted_params)


min_g_factor = -10
max_g_factor = 35
g_factor_observed = 5

x_values = np.linspace(min_g_factor, max_g_factor, 512)
individual_sigmoids = [sigmoid(x_values, slope, shift) for slope, shift in params]
y_sigmoid_combined = np.mean(individual_sigmoids, axis=0)
mask = x_values <= g_factor_observed
fitted_params = get_lognormal_cdf_fit_params(x_values[mask], y_sigmoid_combined[mask])
print("Loc", fitted_params[0], "Sigma", fitted_params[1], "Mu", fitted_params[2])

fig = plt.figure(figsize=(12, 10), constrained_layout=True)
gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.07)

ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()
colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(individual_sigmoids)))
for i, lognormal_cdf_values in enumerate(individual_sigmoids):
    ax2.plot(x_values, lognormal_cdf_values, color=colors[i], alpha=0.4)

train_mask = x_values <= g_factor_observed
ax1.plot(
    x_values[train_mask],
    y_sigmoid_combined[train_mask],
    label="Average Solve Rate (Observed Models)",
    color="blue",
    linewidth=2,
)
ax1.plot(
    x_values[~train_mask],
    y_sigmoid_combined[~train_mask],
    label="Average Solve Rate (Future Models)",
    color="green",
    linewidth=2,
)

lognormal_cdf_values = lognormal_cdf(x_values, *fitted_params)
ax1.plot(x_values, lognormal_cdf_values, "k--", label="Predicted Sigmoid")

# x_target = find_x_quantile(x_values, lognormal_cdf_values, upper_quantile)
# ax1.axvline(
#     x=x_target,
#     color="blue",
#     linestyle="--",
#     label=f"Predicted G-factor with 90% Solve Rate = {x_target:.2f}",
# )
# actual_90 = find_x_quantile(x_values, combined_sigmoid, 0.9)
# ax1.axvline(
#     x=actual_90,
#     color="red",
#     linestyle="--",
#     label=f"Actual G-factor with 90% Solve Rate = {actual_90}",
# )

ax1.set_ylabel("Overall Solve Rate")
ax2.set_ylabel("Individual Task Solve Rate")

ax1.set_ylim(0, 1)
ax2.set_ylim(0, 5)

ax2.spines["right"].set_bounds(0, 1)
ax2.set_yticks(np.linspace(0, 1, 6))

def format_ticks(value, pos):
    if 0 <= value <= 1:
        return f"{value:g}"
    else:
        return ""

ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax2.yaxis.set_label_coords(1.05, 0.15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title(f"Sigmoid Functions and Combined Plot (N tasks = {len(individual_sigmoids)})")

ax1.grid(True)

ax_hist = fig.add_subplot(gs[1], sharex=ax1)
ax_hist.hist(
    [param[1] for param in params], bins=20, color="skyblue", edgecolor="black"
)
ax_hist.set_xlabel("G-factor")
ax_hist.set_ylabel("Frequency of Sigmoids")

plt.setp(ax1.get_xticklabels(), visible=False)

plt.show()

