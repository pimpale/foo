# %%

import matplotlib.pyplot as plt
import numpy as np
import duckdb

tasks = duckdb.read_csv("./tasks.csv")

cybench_ttfs = duckdb.sql(
    """
    SELECT Minutes
    FROM tasks
    WHERE "Task Suite" = 'cybench' and "Included in Large Suite" = 'Yes'
    """
).df()["Minutes"]

fig, ax = plt.subplots(1, 2, figsize=(10, 3))

fig.suptitle("Cybench")

ax1, ax2 = ax

ax1.hist(cybench_ttfs, bins=10, density=True)
ax1.set_title("Time to finish tasks")
ax1.set_xlabel("Time to finish (minutes)")
ax1.set_ylabel("Proportion of tasks")


ax2.hist(np.log(cybench_ttfs), bins=10, density=True)
ax2.set_title("Log time to finish tasks")
ax2.set_xlabel("Log time to finish (minutes)")
ax2.set_ylabel("Proportion of tasks")

gaia_ttfs = duckdb.sql(
    """
    SELECT Minutes
    FROM tasks
    WHERE "Task Suite" = 'gaia' and "Included in Large Suite" = 'Yes'
    """
).df()["Minutes"]

print("Gaia Tasks", len(gaia_ttfs))

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle("Gaia")
ax1, ax2 = ax

ax1.hist(gaia_ttfs, bins=10, density=True)
ax1.set_title("Time to finish tasks")
ax1.set_xlabel("Time to finish (minutes)")
ax1.set_ylabel("Proportion of tasks")


ax2.hist(np.log(gaia_ttfs), bins=10, density=True)
ax2.set_title("Log time to finish tasks")
ax2.set_xlabel("Log time to finish (minutes)")
ax2.set_ylabel("Proportion of tasks")



metr_public_task_ttfs = duckdb.sql(
    """
    SELECT Minutes
    FROM tasks
    WHERE "Task Suite" = 'metr' and "Included in Large Suite" = 'Yes'
    """
).df()["Minutes"]

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle("Metr public task")
ax1, ax2 = ax

ax1.hist(metr_public_task_ttfs, bins=10, density=True)
ax1.set_title("Time to finish tasks")
ax1.set_xlabel("Time to finish (minutes)")
ax1.set_ylabel("Proportion of tasks")

ax2.hist(np.log(metr_public_task_ttfs), bins=10, density=True)
ax2.set_title("Log time to finish tasks")
ax2.set_xlabel("Log time to finish (minutes)")
ax2.set_ylabel("Proportion of tasks")

print("METR Public Tasks", len(metr_public_task_ttfs))

all_task_ttfs = duckdb.sql(
    """
    SELECT Minutes
    FROM tasks
    WHERE "Included in Large Suite" = 'Yes'
    """
).df()["Minutes"]

print("All Tasks", len(all_task_ttfs))
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
fig.suptitle("All tasks")
ax1, ax2 = ax

ax1.hist(all_task_ttfs, bins=10, density=True)
ax1.set_title("Time to finish tasks")
ax1.set_xlabel("Time to finish (minutes)")
ax1.set_ylabel("Proportion of tasks")

ax2.hist(np.log(all_task_ttfs), bins=10, density=True)
ax2.set_title("Log time to finish tasks")
ax2.set_xlabel("Log time to finish (minutes)")
ax2.set_ylabel("Proportion of tasks")
