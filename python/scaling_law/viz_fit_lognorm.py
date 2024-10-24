#%%

from typing import Literal
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from scipy.stats import lognorm
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact
import ipywidgets as widgets



def generate_sigmoid_parameters(
    n_tasks: int,
    slope_range: tuple[float, float],
    shift_range: tuple[float, float],
    distribution: Literal["uniform", "normal", "lognormal"],
    mean_shift: float = 0,
    std_shift: float = 1,
) -> list[tuple[float, float]]:
    k_values = np.random.uniform(*slope_range, n_tasks)
    if distribution == "uniform":
        x0_values = np.random.uniform(*shift_range, n_tasks)
    elif distribution == "normal":
        x0_values = np.clip(
            np.random.normal(mean_shift, std_shift, n_tasks), *shift_range
        )
    elif distribution == "lognormal":
        x0_values = np.clip(
            np.random.lognormal(mean_shift, std_shift, n_tasks), *(shift_range[0], None) 
        )
    else:
        raise ValueError("Distribution must be 'uniform', 'gaussian', or 'lognormal'")
    return list(zip(k_values, x0_values, strict=True))

def sigmoid(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - shift)))

def lognormal_cdf(x, loc, sigma, mu):
    return lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu))

def lognormal_pdf(x, loc, sigma, mu):
    return lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu))

def get_lognormal_cdf_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = curve_fit(
        lognormal_cdf,
        x_values,
        y_values,
        p0=[
            # center at the median
            -10, 0.4, 2.6
        ],
        bounds=([-100, 0.01, 0.1], [20, 10, 10]),
        maxfev=5000
    )
    return popt


#%%

params = generate_sigmoid_parameters(
    # n tasks
    1000,
    # slope range
    # (0.1, 2),
    (10, 11),
    #shift_range,
    (0, 20),
    # distribution
    "lognormal",
    #mean_shift_lognormal,
    1.6,
    #std_shift_lognormal,
    0.7
)

min_g_factor = -10
max_g_factor = 35
g_factor_observed = 5

x_values = np.linspace(min_g_factor, max_g_factor, 512)
individual_sigmoids = [sigmoid(x_values, slope, shift) for slope, shift in params]
y_sigmoid_combined = np.mean(individual_sigmoids, axis=0)
mask = x_values <= g_factor_observed
fitted_params = get_lognormal_cdf_fit_params(x_values[mask], y_sigmoid_combined[mask])
print("Loc", fitted_params[0], "Sigma", fitted_params[1], "Mu", fitted_params[2])

fig = plt.figure(figsize=(12, 10), constrained_layout=True)
gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.07)

ax1 = fig.add_subplot(gs[0])
ax2 = ax1.twinx()
colors = plt.cm.Greys(np.linspace(0.2, 0.8, len(individual_sigmoids)))
for i, lognormal_cdf_values in enumerate(individual_sigmoids):
    ax2.plot(x_values, lognormal_cdf_values, color=colors[i], alpha=0.4)

train_mask = x_values <= g_factor_observed
ax1.plot(
    x_values[train_mask],
    y_sigmoid_combined[train_mask],
    label="Average Solve Rate (Observed Models)",
    color="blue",
    linewidth=2,
)
ax1.plot(
    x_values[~train_mask],
    y_sigmoid_combined[~train_mask],
    label="Average Solve Rate (Future Models)",
    color="green",
    linewidth=2,
)

lognormal_cdf_values = lognormal_cdf(x_values, *fitted_params)
ax1.plot(x_values, lognormal_cdf_values, "k--", label="Predicted Sigmoid")

# x_target = find_x_quantile(x_values, lognormal_cdf_values, upper_quantile)
# ax1.axvline(
#     x=x_target,
#     color="blue",
#     linestyle="--",
#     label=f"Predicted G-factor with 90% Solve Rate = {x_target:.2f}",
# )
# actual_90 = find_x_quantile(x_values, combined_sigmoid, 0.9)
# ax1.axvline(
#     x=actual_90,
#     color="red",
#     linestyle="--",
#     label=f"Actual G-factor with 90% Solve Rate = {actual_90}",
# )

ax1.set_ylabel("Overall Solve Rate")
ax2.set_ylabel("Individual Task Solve Rate")

ax1.set_ylim(0, 1)
ax2.set_ylim(0, 5)

ax2.spines["right"].set_bounds(0, 1)
ax2.set_yticks(np.linspace(0, 1, 6))

def format_ticks(value, pos):
    if 0 <= value <= 1:
        return f"{value:g}"
    else:
        return ""

ax2.yaxis.set_major_formatter(FuncFormatter(format_ticks))
ax2.yaxis.set_label_coords(1.05, 0.15)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

ax1.set_title(f"Sigmoid Functions and Combined Plot (N tasks = {len(individual_sigmoids)})")

ax1.grid(True)

ax_hist = fig.add_subplot(gs[1], sharex=ax1)
ax_hist.hist(
    [param[1] for param in params], bins=20, color="skyblue", edgecolor="black"
)
ax_hist.set_xlabel("G-factor")
ax_hist.set_ylabel("Frequency of Sigmoids")

plt.setp(ax1.get_xticklabels(), visible=False)

plt.show()

#%%

# try to manually fit a lognormal cdf to the combined sigmoid
@interact(
        loc=widgets.FloatSlider(min=-10, max=10, value=0),
        sigma=widgets.FloatSlider(min=0.1, max=10, value=1),
        mu=widgets.FloatSlider(min=-10, max=10, value=1),
)
def g(loc, sigma, mu):
    x = np.linspace(-10, 30, 512)
    plt.plot(x, lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu)))
    plt.plot(x, lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu)))
    plt.plot(x_values, y_sigmoid_combined)
    plt.ylim(0, 1)
    plt.show()
    
#%%

# Plot the pdf of the combined_sigmoid (take the derivative of combined_sigmoid)
# and compare it to the lognormal pdf

def sigmoid_derivative(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return slope * np.exp(-slope * (x - shift)) / (1 + np.exp(-slope * (x - shift)))**2

individual_sigmoid_pdfs = [sigmoid_derivative(x_values, slope, shift) for slope, shift in params]
y_sigmoid_combined_pdf = np.mean(individual_sigmoid_pdfs, axis=0)

fitted_params_pdf_y = lognormal_pdf(x_values, *fitted_params)

plt.plot(x_values, y_sigmoid_combined_pdf, label="Combined Sigmoid PDF")
plt.plot(x_values, fitted_params_pdf_y, label="Fitted Lognormal PDF")
plt.axvline(g_factor_observed, color="red", label="Observed G-factor")
plt.legend()
plt.show()
