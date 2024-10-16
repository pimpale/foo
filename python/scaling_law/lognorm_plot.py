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

# compute average over tasks for each model
model_avgs = defaultdict(lambda: {'pc1': [], 'avg': []})
for task, model_scores in results_dict_avg_std.items():
    for model, scores in model_scores.items():
        model_avgs[model]['pc1'].append(scores['pc1'])
        model_avgs[model]['avg'].append(scores['avg'])


model_avg = {}
for model, scores in model_avgs.items():
    model_avg[model] = {
        'pc1': np.mean(scores['pc1']),
        'avg': np.mean(scores['avg'])
    }


sorted_models = sorted(model_avgs.keys(), key=lambda x: model_avg[x]['pc1'])


# scatter plot of each model's PC1 score vs. average success rate (labeled)

fig, ax = plt.subplots()
ax.set_xlabel('PC1')
ax.set_ylabel('Average Success Rate')
for model in sorted_models:
    ax.scatter(model_avg[model]['pc1'], model_avg[model]['avg'], label=model)


# fit a sigmoid to the data
x_values = np.array([model_avg[model]['pc1'] for model in sorted_models])
y_values = np.array([model_avg[model]['avg'] for model in sorted_models])
params = get_sigmoid_parameters(x_values, y_values)
print("Slope", params[0], "Shift", params[1])
x_values = np.linspace(0, 10, 512)
y_sigmoid = sigmoid(x_values, *params)
ax.plot(x_values, y_sigmoid, label="Sigmoid Fit", color="red")

# fit a lognormal to the data
params = get_lognormal_cdf_fit_params(x_values, y_sigmoid)
print("Loc", params[0], "Sigma", params[1], "Mu", params[2])
y_lognormal = lognormal_cdf(x_values, *params)
ax.plot(x_values, y_lognormal, label="Lognormal Fit", color="green")

ax.legend()
plt.show()