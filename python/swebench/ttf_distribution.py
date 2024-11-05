# %%

import duckdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

eap = duckdb.read_csv("./ensembled_annotations_public.csv")
sw3ap = pd.read_csv('./samples_with_3_annotations_public.csv')

difficulty_to_minutes = {
    "<15 min fix": 7.5,
    "15 min - 1 hour": 37.5,
    "1-4 hours": 150,
    ">4 hours": 300,
}

difficulty_to_minutes_df = pd.DataFrame(
    difficulty_to_minutes.items(), columns=["difficulty", "difficulty_in_minutes"]
)

task_difficulties = duckdb.sql(
    """
    SELECT eap.instance_id, dtm.difficulty_in_minutes
    FROM eap
    JOIN difficulty_to_minutes_df as dtm ON eap.difficulty = dtm.difficulty
    WHERE not(filter_out)
    """
).to_df()

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
fig.suptitle("SWEBench Verified")

# plot non-log scale
ax[0, 0].hist(task_difficulties["difficulty_in_minutes"], bins=20)
ax[0, 0].set_xlabel("Time to fix (minutes)")
ax[0, 0].set_ylabel("Number of tasks")
ax[0, 0].set_title("Time to fix ensembled")

# plot log scale
ax[0, 1].hist(np.log(task_difficulties["difficulty_in_minutes"]), bins=20)
ax[0, 1].set_xlabel("Log time to fix (minutes)")
ax[0, 1].set_ylabel("Number of tasks")
ax[0, 1].set_title("Log time to fix ensembled")
ax[0, 1].set_xlim(0, 6)

task_difficulties_annotator_avg = duckdb.sql(
    """
    WITH valid_annotations as (
        SELECT sw3ap.instance_id, sw3ap.difficulty
        FROM sw3ap
        JOIN eap ON sw3ap.instance_id = eap.instance_id
        WHERE not(filter_out)
    )
    SELECT valid_annotations.instance_id, AVG(dtm.difficulty_in_minutes) as avg_difficulty_in_minutes
    FROM valid_annotations
    JOIN difficulty_to_minutes_df as dtm ON valid_annotations.difficulty = dtm.difficulty
    GROUP BY valid_annotations.instance_id
    """
).to_df()

# plot non-log scale
ax[1, 0].hist(task_difficulties_annotator_avg["avg_difficulty_in_minutes"], bins=20)
ax[1, 0].set_xlabel("Average time to fix (minutes)")
ax[1, 0].set_ylabel("Number of tasks")
ax[1, 0].set_title("Average time to fix")

# plot log scale
ax[1, 1].hist(np.log(task_difficulties_annotator_avg["avg_difficulty_in_minutes"]), bins=20)
ax[1, 1].set_xlabel("Log average time to fix (minutes)")
ax[1, 1].set_ylabel("Number of tasks")
ax[1, 1].set_title("Log average time to fix")
ax[1, 1].set_xlim(0, 6)
