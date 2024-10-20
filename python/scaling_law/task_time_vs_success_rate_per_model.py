# %%

import numpy as np
import scipy.stats as stats
import pandas as pd
import duckdb
from matplotlib import pyplot as plt
from collections import defaultdict
import scipy.optimize as opt


results = duckdb.read_csv("./results.csv")
tasks = duckdb.read_csv("./tasks.csv")
pc1_scores = duckdb.read_csv("./leaderboard_pca_scores.csv")

# for each model:
# graph the time each task takes on the x-axis and the success rate on the y-axis


modelTaskResultsMinute = duckdb.sql(
    """
    WITH 
    filteredResults AS (
        SELECT score, taskFamily, taskName, model
        FROM results
        WHERE score != -1
    ),
    modelTaskResults AS (
        SELECT AVG(score) as avg_score, taskFamily, taskName,  model
        FROM filteredResults
        GROUP BY taskFamily, taskName, model
    )
    SELECT model, taskFamily, taskName, avg_score, Minutes
    FROM modelTaskResults
    JOIN tasks ON modelTaskResults.taskFamily = tasks."Task Suite" AND modelTaskResults.taskName = tasks."Task Name"
    """
).df()


models = [
    model 
    for (model,) 
    in duckdb.sql(
        """
        SELECT model from modelTaskResultsMinute
        GROUP BY model
        HAVING COUNT(*) == 34 
        """
    ).fetchall()
]

ncols = 3
nrows = len(models) // ncols + 1


# exponential decay function (with x offset)
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * (x - c))


fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
axs = axs.flatten()
for model, ax in zip(models, axs):
    modelTaskResultsMinuteModel = modelTaskResultsMinute[modelTaskResultsMinute["model"] == model]
    print(len(modelTaskResultsMinuteModel), model)
    print(modelTaskResultsMinuteModel)
    ax.scatter(modelTaskResultsMinuteModel["Minutes"], modelTaskResultsMinuteModel["avg_score"], label="Task")
    # Fit a line 
    slope, intercept, r_value, p_value, std_err = stats.linregress(modelTaskResultsMinuteModel["Minutes"], modelTaskResultsMinuteModel["avg_score"])
    line = slope * modelTaskResultsMinuteModel["Minutes"] + intercept
    ax.plot(modelTaskResultsMinuteModel["Minutes"], line, 'r', label='y={:.2f}x+{:.2f}'.format(slope,intercept))
    
    # now try fitting an exponential decay function
    best_params = opt.curve_fit(
        exponential_decay, 
        modelTaskResultsMinuteModel["Minutes"], 
        modelTaskResultsMinuteModel["avg_score"], 
        p0=[1, 0.1, 0]
    )[0]
    print(best_params)
    lin = np.linspace(0, max(modelTaskResultsMinute['Minutes']), 1000)
    ax.plot(lin, exponential_decay(lin, *best_params), label="Exponential Decay Fit", color="orange")
    
    ax.set_title(model)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1)    
    ax.legend()
    

