#%%

import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt

bins = np.array([
    [1, 4],
    [4, 15],
    [15, 1*60],
    [1*60, 4*60],
    [4*60, 16*60],
    [16*60, 64*60],
])

pSuccess = np.array([
    0.55,
    0.52,
    0.28,
    0.24,
    0.10,
    0.005,
])

colors =["blue", "orange", "green", "red", "cyan", "purple"]

# Create a bar plot with the bins on the x-axis and the success rate on the y-axis (not to scale, 1 bar per bin)

fig, ax = plt.subplots(2, 1, figsize=(10, 10))
ax1, ax2 = ax
ax1.bar(
    [f"{low}-{high} min" for low, high in bins],
    pSuccess,
    color=colors
)
ax1.set_ylabel("Success Rate")
ax1.set_xlabel("Time to Solve")
ax1.set_title("Success Rate vs. Time to Solve")
ax1.set_ylim(0, 1)



# Create a histogram with the bins on the x-axis and the success rate on the y-axis (to scale, 1 bar per bin)

for low, high, p, color in zip(bins[:,0], bins[:,1], pSuccess, colors):
    ax2.bar(low, p, width=high-low, align="edge", color=color)


# ax2.bar(bins[:,0], pSuccess, width=bins[:,1]-bins[:,0], align="edge")
ax2.set_ylabel("Success Rate")
ax2.set_xlabel("Time to Solve")
ax2.set_title("Success Rate vs. Time to Solve")
ax2.set_ylim(0, 1)

# exponential decay function (with x offset)
def exponential_decay(x, a, b, c):
    return a * np.exp(-b * (x - c))

best_params = opt.curve_fit(
    exponential_decay, 
    bins.mean(axis=-1), 
    pSuccess, 
    p0=[1, 0.1, 0]
)[0]

print(best_params)

lin = np.linspace(0, 64*60, 1000)

# ax2.plot(lin, exponential_decay(lin, *best_params), label="Exponential Decay Fit", color="red")

# for i in range(10):
#     print(best_params)
#     # Sample 