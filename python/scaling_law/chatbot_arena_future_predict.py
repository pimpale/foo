# %%

import numpy as np
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from ipywidgets import interact
import ipywidgets as widgets


def sigmoid(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - shift)))


def lognormal_cdf(x, loc, sigma, mu):
    return lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu))


def lognormal_pdf(x, loc, sigma, mu):
    return lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu))


def get_sigmoid_parameters(
    x_values: np.ndarray,
    y_values: np.ndarray,
    p0: tuple[
        # slope
        float,
        # shift
        float,
    ],
) -> tuple[float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        sigmoid, x_values, y_values, p0=p0, bounds=([0, 1000], [10, 2300]), maxfev=5000
    )
    return popt


def get_lognormal_cdf_fit_params(
    x_values: np.ndarray,
    y_values: np.ndarray,
    p0: tuple[
        # loc
        float,
        # mu
        float,
        # sigma
        float,
    ],
) -> tuple[float, float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        lognormal_cdf,
        x_values,
        y_values,
        p0=p0,
        bounds=([900, 0.01, 0.1], [2024, 10, 10]),
        maxfev=5000,
    )
    return popt


model_name_mapping = duckdb.read_csv("./data_models/meta/model_name_mapping.csv")

chatbot_arena_scores = duckdb.read_csv("./data_models/cache_new/chatbot_arena.csv")

chatbot_arena_release_dates = duckdb.read_csv(
    "./data_models/meta/chatbot_arena_release_dates.csv"
)

# scatter plot of chatbot arena scores
msrd = duckdb.sql(
    """
    SELECT 
        cas.Model as model, 
        cas."Arena Score" as score, 
        (year(card.release_date) + (1/365)*dayofyear(card.release_date)) as release_date
    FROM chatbot_arena_scores cas
    JOIN chatbot_arena_release_dates card ON card.Model = cas.Model
    """
).df()

plt.scatter(msrd["release_date"], msrd["score"])

# create x that can be used for plotting functions
x = np.linspace(min(msrd["release_date"]), max(msrd["release_date"]) + 1, 100)

# table of models that were ever at the top 5 of the chatbot arena
top_n_results = duckdb.sql(
    """
    SELECT model, score, release_date
    FROM msrd
    WHERE score >= (
        SELECT msrd2.score
        FROM msrd as msrd2
        WHERE msrd2.release_date <= msrd.release_date
        ORDER BY msrd2.score DESC
        LIMIT 1 OFFSET 5
    )
    """
).df()

plt.scatter(top_n_results["release_date"], top_n_results["score"], label="top 5")

top5_params = stats.linregress(top_n_results["release_date"], top_n_results["score"])
print(top5_params)

# plot line
plt.plot(
    x,
    top5_params.intercept + top5_params.slope * x,
    color="red",
)


# table of models that were ever at the top of the chatbot arena
best_results = duckdb.sql(
    """
    SELECT model, score, release_date
    FROM msrd
    WHERE score >= (
        SELECT msrd2.score
        FROM msrd as msrd2
        WHERE msrd2.release_date <= msrd.release_date
        ORDER BY msrd2.score DESC
        LIMIT 1
    )
    """
).df()

plt.scatter(
    best_results["release_date"], best_results["score"], label="best", color="green"
)

# drop llama 13b because it is an outlier
best_results = best_results[best_results["score"] > 1000]

top1_params = stats.linregress(best_results["release_date"], best_results["score"])
print(top1_params)

# plot line
plt.plot(
    x,
    top1_params.intercept + top1_params.slope * x,
    color="green",
)

plt.legend()
plt.show()


# create a seperate plot where we fit the task results
trials = duckdb.read_csv("./data_models/trials/joint.csv")
model_name_mapping = duckdb.read_csv("./data_models/meta/model_name_mapping.csv")

# filter out the -1 scores
# and rename the models to what they would be in the chatbot arena
trials_filtered = duckdb.sql(
    """
    SELECT score, concat(task_family, ' ', task_name) as task, mnm.chatbot_arena_name as model
    FROM trials as t
    JOIN model_name_mapping as mnm ON t.model = mnm.model_name
    WHERE score != -1
    """
).df()

# average over the tasks
tasks = duckdb.sql(
    """
    SELECT model, avg(score) as score
    FROM trials_filtered
    GROUP BY model, task
    """
).df()

success_rates = duckdb.sql(
    """
    SELECT model, avg(score) as success_rate, 
    FROM tasks
    GROUP BY model
    """
).df()

elo_scores = duckdb.sql(
    """
    SELECT msrd.model, success_rate, score, release_date
    FROM success_rates
    JOIN msrd ON success_rates.model = msrd.model
    """
).df()


# scatter plot of each model's PC1 score vs. average success rate (labeled)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax1, ax2 = ax

ax1.set_xlabel("ELO Score")
ax1.set_ylabel("Average Success Rate")
ax1.set_ylim(0, 1)

# Scatter plot of ELO score vs. success rate
ax1.scatter(elo_scores["score"], elo_scores["success_rate"], label="success rate")

# fit a sigmoid to the data
x_values = np.array(elo_scores["score"])
y_values = np.array(elo_scores["success_rate"])
params = get_sigmoid_parameters(x_values, y_values, [1, 1200])
print("Slope", params[0], "Shift", params[1])
x_linspace = np.linspace(1000, 1600, 512)
y_sigmoid = sigmoid(x_linspace, *params)
ax1.plot(x_linspace, y_sigmoid, label="Sigmoid Fit", color="red")

# fit a lognormal to the data
params = get_lognormal_cdf_fit_params(x_values, y_values, [1000, 0.4, 2.6])
print("Loc", params[0], "Sigma", params[1], "Mu", params[2])
y_lognormal = lognormal_cdf(x_linspace, *params)
ax1.plot(x_linspace, y_lognormal, label="Lognormal Fit", color="green")


ax1.legend()

# Try a linear transformation from ELO to year, then graph the success rate over time

top5_score_2_releasedate = stats.linregress(
    top_n_results["score"], top_n_results["release_date"]
)

x_values = (
    np.array(elo_scores["score"]) * top5_score_2_releasedate.slope
    + top5_score_2_releasedate.intercept
)
y_values = np.array(elo_scores["success_rate"])

# Scatter plot of ELO score vs. success rate
ax2.scatter(x_values, y_values, label="success rate")

ax2.set_xlabel("Year")
ax2.set_ylabel("Average Success Rate")
ax2.set_ylim(0, 1)

# fit a sigmoid to the data
params = get_sigmoid_parameters(x_values, y_values, [1, 2022])
print("Slope", params[0], "Shift", params[1])
x_linspace = np.linspace(2022, 2025.5, 512)
y_sigmoid = sigmoid(x_linspace, *params)
ax2.plot(x_linspace, y_sigmoid, label="Sigmoid Fit", color="red")

# fit a lognormal to the data
params = get_lognormal_cdf_fit_params(x_values, y_values, [2022, 0.4, 2.6])
print("Loc", params[0], "Sigma", params[1], "Mu", params[2])
y_lognormal = lognormal_cdf(x_linspace, *params)
ax2.plot(x_linspace, y_lognormal, label="Lognormal Fit", color="green")

ax2.legend()

#%% 

# Try to figure out why the lognormal fit is so bad

@interact(
    loc=widgets.FloatSlider(min=2000, max=2025, step=0.1, value=2022),
    sigma=widgets.FloatSlider(min=0, max=1, step=0.01, value=0.4),
    mu=widgets.FloatSlider(min=0, max=10, step=0.1, value=2.6),
)
def g(slope, shift, loc, sigma, mu):
    x_values = (
        np.array(elo_scores["score"]) * top5_score_2_releasedate.slope
        + top5_score_2_releasedate.intercept
    )
    y_values = np.array(elo_scores["success_rate"])

    # Scatter plot of ELO score vs. success rate
    plt.scatter(x_values, y_values, label="success rate")

    plt.xlabel("Year")
    plt.ylabel("Average Success Rate")
    plt.ylim(0, 1)

    # fit a sigmoid to the data
    params = get_sigmoid_parameters(x_values, y_values, [slope, shift])
    print("Slope", params[0], "Shift", params[1])
    x_linspace = np.linspace(2022, 2025.5, 512)
    y_sigmoid = sigmoid(x_linspace, *params)
    plt.plot(x_linspace, y_sigmoid, label="Sigmoid Fit", color="red")

    # fit a lognormal to the data
    params = get_lognormal_cdf_fit_params(x_values, y_values, [loc, sigma, mu])
    print("Loc", params[0], "Sigma", params[1], "Mu", params[2])
    y_lognormal = lognormal_cdf(x_linspace, *params)
    plt.plot(x_linspace, y_lognormal, label="Lognormal Fit", color="green")

    plt.legend()
    plt.show()