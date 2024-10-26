#%%
import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
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
from utils.plot_markers import markers


tasks = duckdb.read_csv("./tasks.csv")

all_task_ttfs = duckdb.sql(
    """
    SELECT concat("Task Suite", ' ', "Task Name") as task, Minutes as minutes
    FROM tasks
    WHERE "Included in Large Suite" = 'Yes'
    """
).df()

log_task_ttfs = np.log(all_task_ttfs["minutes"])

# fit a normal distribution to the non-log data
p_mu, p_std = stats.norm.fit(all_task_ttfs["minutes"])

# fit a normal distribution to the log data
q_mu, q_std = stats.norm.fit(log_task_ttfs)

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle("All tasks")
ax1, ax2 = ax

ax1.hist(all_task_ttfs["minutes"], bins=10, density=True)
ax1.set_title("Time to finish tasks")
ax1.set_xlabel("Time to finish (minutes)")
ax1.set_ylabel("Proportion of tasks")

x_linspace = np.linspace(min(all_task_ttfs["minutes"]), max(all_task_ttfs["minutes"]), 1000)
ax1.plot(x_linspace, stats.norm.pdf(x_linspace, p_mu, p_std))
ax1.plot(x_linspace, stats.norm.pdf(np.log(x_linspace), q_mu, q_std))

ax2.hist(log_task_ttfs, bins=10, density=True)
ax2.set_title("Log time to finish tasks")
ax2.set_xlabel("Log time to finish (minutes)")
ax2.set_ylabel("Proportion of tasks")

x_linspace = np.linspace(min(log_task_ttfs), max(log_task_ttfs), 1000)
ax2.plot(x_linspace, stats.norm.pdf(np.exp(x_linspace), p_mu, p_std)*20)
ax2.plot(x_linspace, stats.norm.pdf(x_linspace, q_mu, q_std))

plt.show()

#%%

# compute p(x) for all tasks

# But first, note that we need to account for the fact that p(x) has infinite support
extra_mass = stats.norm.cdf(min(log_task_ttfs), mu, std) + (1 - stats.norm.cdf(max(log_task_ttfs), mu, std))

px = stats.norm.pdf(log_task_ttfs, mu, std) * (1+extra_mass)

# create histogram of log_task_ttfs using numpy (use 10 bdins)
q_pmf, bins = np.histogram(log_task_ttfs, bins=10, density=True)
qx = q_pmf[np.digitize(log_task_ttfs, bins, right=True) - 1]

wx = px / qx

task_ttfs_weights = all_task_ttfs.copy().assign(weight=wx)

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
        bounds=([800, 0.01, 0.1], [2024, 10, 10]),
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

# average over the tasks
tasks_weighted = duckdb.sql(
    """
    SELECT model, avg(score*weight) as score
    FROM trials_filtered
    JOIN task_ttfs_weights as ttfs ON trials_filtered.task = ttfs.task
    GROUP BY model, trials_filtered.task
    """
).df()

def plot_task_success_rate_vs_elo(title, ax, tasks, threshold_elo=1200):

    def compute_mse_sigmoid(x_values, y_values, params):
        y_sigmoid = sigmoid(x_values, *params)
        return np.mean((y_sigmoid - y_values) ** 2)
    
    def compute_mse_lognormal(x_values, y_values, params):
        y_lognormal = lognormal_cdf(x_values, *params)
        return np.mean((y_lognormal - y_values) ** 2)


    elo_scores = duckdb.sql(
        """
        WITH success_rates AS (
            SELECT model, avg(score) as success_rate, 
            FROM tasks
            GROUP BY model
        )
        SELECT msrd.model, success_rate, score, release_date
        FROM success_rates
        JOIN msrd ON success_rates.model = msrd.model
        ORDER BY score ASC
        """
    ).df()  

    ax.set_title(title)
    ax.set_xlabel("ELO Score")
    ax.set_ylabel("Average Success Rate")
    ax.set_ylim(0, 1)

    # Scatter plot of ELO score vs. success rate
    for i, row in elo_scores.iterrows():
        color = "red" if row["score"] > threshold_elo else "blue"
        ax.scatter(row["score"], row["success_rate"], label=row["model"], marker=markers[i], color=color)

    # fit a sigmoid to the data
    x_values = np.array(elo_scores[elo_scores["score"] <= threshold_elo]["score"])
    y_values = np.array(elo_scores[elo_scores["score"] <= threshold_elo]["success_rate"])

    x_values_test = np.array(elo_scores[elo_scores["score"] > threshold_elo]["score"])
    y_values_test = np.array(elo_scores[elo_scores["score"] > threshold_elo]["success_rate"])

    params = get_sigmoid_parameters(x_values, y_values, [1, 1100])
    print("Slope", params[0], "Shift", params[1])
    x_linspace = np.linspace(1000, 1600, 512)
    y_sigmoid = sigmoid(x_linspace, *params)
    ax.plot(x_linspace, y_sigmoid, label="Sigmoid Fit", color="red")

    # compute the MSE of the test data
    train_mse_sigmoid = compute_mse_sigmoid(x_values, y_values, params)
    test_mse_sigmoid = compute_mse_sigmoid(x_values_test, y_values_test, params)
    ax.text(0.1, 0.95, f"[train] Sigmoid MSE: {train_mse_sigmoid:.2e}", transform=ax.transAxes)
    ax.text(0.1, 0.9, f"[test] Sigmoid MSE: {test_mse_sigmoid:.2e}", transform=ax.transAxes)

    # fit a lognormal to the data
    params = get_lognormal_cdf_fit_params(x_values, y_values, [1100, 0.7, 2.6])
    print("Loc", params[0], "Sigma", params[1], "Mu", params[2])
    y_lognormal = lognormal_cdf(x_linspace, *params)
    ax.plot(x_linspace, y_lognormal, label="Lognormal Fit", color="green")

    # compute the MSE of the test data
    train_mse_lognormal = compute_mse_lognormal(x_values, y_values, params)
    test_mse_lognormal = compute_mse_lognormal(x_values_test, y_values_test, params)
    ax.text(0.1, 0.85, f"[train] Lognormal MSE: {train_mse_lognormal:.2e}", transform=ax.transAxes)
    ax.text(0.1, 0.8, f"[test] Lognormal MSE: {test_mse_lognormal:.2e}", transform=ax.transAxes)

# scatter plot of each model's PC1 score vs. average success rate (labeled)


fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax1, ax2 = ax

plot_task_success_rate_vs_elo("Unweighted Tasks", ax1, tasks)

fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)

plot_task_success_rate_vs_elo("Weighted Tasks (Importance Sampled)", ax2, tasks_weighted)


