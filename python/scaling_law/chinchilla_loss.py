# %%
import numpy as np
import matplotlib.pyplot as plt
import util_plot_markers


# Define the Chinchilla scaling law loss function
# L = E + A/N^alpha + B/D^beta
alpha = 0.34
beta = 0.28
A = 406.4
B = 410.7
E = 1.69

# Define table of optimal chinchilla parameters described in Approach 3 of the paper:
approach_3 = [
    # [400e6, 9.2e9],
    [1e9, 27.1e9],
    [10e9, 410.1e9],
    [67e9, 4.1e12],
    [175e9, 12.0e12],
    [280e9, 20.1e12],
    [520e9, 43.5e12],
]

# define table of known models
models = [
    [2.6146368 * 1e9, 2 * 1e12, "Gemma 2 2B"],
    [9.242164736 * 1e9, 8 * 1e12, "Gemma 2 9B"],
    [27.22771814 * 1e9, 13 * 1e12, "Gemma 2 27B"],
    [2.50643456 * 1e9, 3 * 1e12, "Gemma 1.1 2B"],
    [8.538074112 * 1e9, 6 * 1e12, "Gemma 1.1 7B"],
    [2.50643456 * 1e9, 2 * 1e12, "Gemma 2B"],
    [8.538074112 * 1e9, 6 * 1e12, "Gemma 7B"],
]


def loss(n, d) -> float:
    return E + A / n**alpha + B / d**beta


def opt_params(L_budget: float) -> tuple[float, float]:
    l = L_budget - E
    N_opt = (A * (alpha + beta) / (l * beta)) ** (1 / alpha)
    D_opt = (B * (alpha + beta) / (l * alpha)) ** (1 / beta)
    return N_opt, D_opt


# Define the ranges for N (number of parameters) and D (number of data tokens)
N_values = np.logspace(9, 12, 100)  # From 1e9 to 1e12
D_values = np.logspace(9, 15, 100)  # From 1e9 to 1e15

N, D = np.meshgrid(N_values, D_values)

Loss = E + A / N**alpha + B / D**beta
C = 6 * N * D

fig, ax = plt.subplots(1, 1, figsize=(8, 7))

# Set the plot labels and title
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of Parameters (N)")
ax.set_ylabel("Number of Data Tokens (D)")
ax.set_title("Chinchilla Scaling Law Contour Plot")
# Plot the chinchilla loss
CS1 = ax.contourf(N, D, Loss, levels=20, cmap="viridis")
fig.colorbar(CS1, label="Loss")

# plot the compute isolines
CS2 = ax.contour(
    N,
    D,
    C,
    levels=list(np.geomspace(C.min(), C.max(), 10)),
    colors="white",
    linestyles="dashed",
)
ax.clabel(CS2, fmt=lambda x: f"{x:2.2e} FLOPs", inline=True, fontsize=7)

# plot optimal points
target_loss_values = np.arange(1.77, 2.5, 0.1)
n_opts, d_opts = zip(*[opt_params(l) for l in target_loss_values])
plt.plot(
    n_opts,
    d_opts,
    label="Chinchilla Optimal Points",
    marker="x",
    color="red",
    linestyle="solid",
)

# plot known models
visualized_loss_curves = []
for i, (n, d, label) in enumerate(models):
    marker = util_plot_markers.markers_scrambled[i]

    l = loss(n, d)
    n_opt, d_opt = opt_params(l)

    # check that l is not too close to any previously visualized loss curves
    if not any(abs(l - l_) < 0.01 for l_ in visualized_loss_curves):
        visualized_loss_curves.append(l)
        # Plot the isoloss curve at L = 2.0
        CS3 = ax.contour(N, D, Loss, levels=[l], colors="yellow")
        ax.clabel(CS3, fmt=f"Loss: {l:2.2}", inline=True, fontsize=7, colors="yellow")

    plt.plot(
        n,
        d,
        marker=marker,
        color="orange",
        linestyle="none",
        label=label,
    )
    plt.plot(
        n_opt,
        d_opt,
        marker=marker,
        linestyle="none",
        color="green",
        label=f"{label} Optimal",
    )


fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.05), ncol=4)

plt.show()
