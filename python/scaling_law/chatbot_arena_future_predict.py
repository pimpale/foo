# %%

import numpy as np
import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import re
from scipy import stats

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
x = np.linspace(min(msrd["release_date"]), max(msrd["release_date"])+1, 100)

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

plt.scatter(top_n_results["release_date"], top_n_results["score"], label='top 5')

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

plt.scatter(best_results["release_date"], best_results["score"], label='best', color='green')

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
#%%

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

plt.scatter(elo_scores["score"], elo_scores["success_rate"], label='success rate')

# fit sigmoid