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
    ORDER BY model, Minutes
    """
).df()


models = [
    model 
    for (model,) 
    in duckdb.sql(
        """
        SELECT model from modelTaskResultsMinute
        GROUP BY model
        HAVING COUNT(*) > 10
        ORDER BY model
        """
    ).fetchall()
]

bins = [
    [0, 4],
    [4, 15],
    [15, 60],
]
    

ncols = 3
nrows = len(models) // ncols + 1

fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*5))
axs = axs.flatten()
for model, ax in zip(models, axs):
    modelTaskResultsMinuteModel = modelTaskResultsMinute[modelTaskResultsMinute["model"] == model]
    print(len(modelTaskResultsMinuteModel), model)
    print(modelTaskResultsMinuteModel)
    # ax.scatter(modelTaskResultsMinuteModel["Minutes"], modelTaskResultsMinuteModel["avg_score"], label="Task")

    for bin_start, bin_end in bins:
        bin = modelTaskResultsMinuteModel[
            (modelTaskResultsMinuteModel["Minutes"] >= bin_start) & (modelTaskResultsMinuteModel["Minutes"] < bin_end)
        ]
        stddev = bin["avg_score"].std()
        print(stddev)
        ax.bar(bin_start, bin["avg_score"].mean(), width=bin_end-bin_start, yerr=stddev, label=f"{bin_start}-{bin_end} minutes", align="edge")
    
    ax.set_title(model)
    ax.set_xlabel("Minutes")
    ax.set_ylabel("Average Score")
    ax.set_ylim(0, 1)
    ax.legend()
    

taskResultsMinute = duckdb.sql(
    """
    SELECT AVG(avg_score) as avg_score, taskFamily, taskName, AVG(Minutes) as Minutes
    FROM modelTaskResultsMinute
    GROUP BY taskFamily, taskName, Minutes
    """
).df()

fig, ax = plt.subplots()
# ax.scatter(taskResultsMinute["Minutes"], taskResultsMinute["avg_score"], label="Task")
for bin_start, bin_end in bins:
    bin = taskResultsMinute[
        (taskResultsMinute["Minutes"] >= bin_start) & (taskResultsMinute["Minutes"] < bin_end)
    ]
    stddev = bin["avg_score"].std()
    print(stddev)
    ax.bar(bin_start, bin["avg_score"].mean(), width=bin_end-bin_start, yerr=stddev, label=f"{bin_start}-{bin_end} minutes", align="edge")

ax.set_title("All Tasks")
ax.set_xlabel("Minutes")
ax.set_ylabel("Average Score")
ax.set_ylim(0, 1)
ax.legend()