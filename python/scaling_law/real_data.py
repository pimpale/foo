#%%

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from collections import defaultdict


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
                'std': np.std(scores)
            }
    results_dict_avg_std[task] = avg_scores    

# create plot for each task, with pc1_score on x-axis and the average success rate on y-axis

for task, model_scores in results_dict_avg_std.items():
    fig, ax = plt.subplots()
    ax.set_title(task)
    ax.set_xlabel('PC1')
    ax.set_ylabel('Average Success Rate')
    for model, scores in model_scores.items():
        ax.errorbar(pc1_scores_dict[model], scores['avg'], yerr=scores['std'], fmt='o', label=model)
    ax.legend()
    plt.show()