import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from matplotlib.ticker import FuncFormatter
from matplotlib.gridspec import GridSpec
from model import (
    sigmoid,
    lognormal_cdf,
    generate_sigmoid_parameters,
    y_sigmoid_combined_function,
    get_individual,
    get_sigmoid_fit_params,
    get_lognormal_cdf_fit_params,
    find_x_quantile,
)


def plot_sigmoid_functions(
    x_values,
    individual_sigmoids,
    combined_sigmoid,
    fitted_params,
    n_tasks,
    scale_factor,
    train_max_g_factor_observed,
    params,
    upper_quantile,
):
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, n_tasks))
    for i, sigmoid_values in enumerate(individual_sigmoids):
        ax2.plot(x_values, sigmoid_values, color=colors[i], alpha=0.4)

    train_mask = x_values <= train_max_g_factor_observed
    ax1.plot(
        x_values[train_mask],
        combined_sigmoid[train_mask],
        label="Average Solve Rate (Observed Models)",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        x_values[~train_mask],
        combined_sigmoid[~train_mask],
        label="Average Solve Rate (Future Models)",
        color="green",
        linewidth=2,
    )

    def sigmoid_fit_curve(x):
        k, x0 = fitted_params
        return sigmoid(x, k, x0)

    sigmoid_values = sigmoid_fit_curve(x_values)
    ax1.plot(x_values, sigmoid_values, "k--", label="Predicted Sigmoid")

    x_target = find_x_quantile(x_values, sigmoid_values, upper_quantile)
    ax1.axvline(
        x=x_target,
        color="blue",
        linestyle="--",
        label=f"Predicted G-factor with 90% Solve Rate = {x_target:.2f}",
    )
    actual_90 = find_x_quantile(x_values, combined_sigmoid, 0.9)
    ax1.axvline(
        x=actual_90,
        color="red",
        linestyle="--",
        label=f"Actual G-factor with 90% Solve Rate = {actual_90}",
    )

    ax1.set_ylabel("Overall Solve Rate")
    ax2.set_ylabel("Individual Task Solve Rate")

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, scale_factor)

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

    ax1.set_title(f"Sigmoid Functions and Combined Plot (N tasks = {n_tasks})")

    ax1.grid(True)

    ax_hist = fig.add_subplot(gs[1], sharex=ax1)
    ax_hist.hist(
        [param[1] for param in params], bins=20, color="skyblue", edgecolor="black"
    )
    ax_hist.set_xlabel("G-factor")
    ax_hist.set_ylabel("Frequency of Sigmoids")

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.show()
    



def plot_sigmoid_functions_lognormal_cdf_fit(
    x_values,
    individual_sigmoids,
    combined_sigmoid,
    fitted_params,
    n_tasks,
    scale_factor,
    train_max_g_factor_observed,
    params,
    upper_quantile,
):
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    gs = GridSpec(2, 1, height_ratios=[8, 1.5], hspace=0.07)

    ax1 = fig.add_subplot(gs[0])
    ax2 = ax1.twinx()
    colors = plt.cm.Greys(np.linspace(0.2, 0.8, n_tasks))
    for i, lognormal_cdf_values in enumerate(individual_sigmoids):
        ax2.plot(x_values, lognormal_cdf_values, color=colors[i], alpha=0.4)

    train_mask = x_values <= train_max_g_factor_observed
    ax1.plot(
        x_values[train_mask],
        combined_sigmoid[train_mask],
        label="Average Solve Rate (Observed Models)",
        color="blue",
        linewidth=2,
    )
    ax1.plot(
        x_values[~train_mask],
        combined_sigmoid[~train_mask],
        label="Average Solve Rate (Future Models)",
        color="green",
        linewidth=2,
    )

    def lognormal_cdf_fit_curve(x):
        shift, s, scale = fitted_params
        return lognormal_cdf(x, shift, s, scale)

    lognormal_cdf_values = lognormal_cdf_fit_curve(x_values)
    ax1.plot(x_values, lognormal_cdf_values, "k--", label="Predicted Sigmoid")

    x_target = find_x_quantile(x_values, lognormal_cdf_values, upper_quantile)
    ax1.axvline(
        x=x_target,
        color="blue",
        linestyle="--",
        label=f"Predicted G-factor with 90% Solve Rate = {x_target:.2f}",
    )
    actual_90 = find_x_quantile(x_values, combined_sigmoid, 0.9)
    ax1.axvline(
        x=actual_90,
        color="red",
        linestyle="--",
        label=f"Actual G-factor with 90% Solve Rate = {actual_90}",
    )

    ax1.set_ylabel("Overall Solve Rate")
    ax2.set_ylabel("Individual Task Solve Rate")

    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, scale_factor)

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

    ax1.set_title(f"Sigmoid Functions and Combined Plot (N tasks = {n_tasks})")

    ax1.grid(True)

    ax_hist = fig.add_subplot(gs[1], sharex=ax1)
    ax_hist.hist(
        [param[1] for param in params], bins=20, color="skyblue", edgecolor="black"
    )
    ax_hist.set_xlabel("G-factor")
    ax_hist.set_ylabel("Frequency of Sigmoids")

    plt.setp(ax1.get_xticklabels(), visible=False)

    plt.show()


def main():
    n_tasks = 100
    scale_factor = 5
    slope_range = (0.1, 2)
    g_factor_cutoff = 5
    shift_range = (0, 20)
    x_start = -10
    x_end = 35
    n_points = 200
    seed = 10
    upper_quantile = 0.9
    mean_shift = 10
    std_shift = 4
    mean_shift_lognormal = 1.6
    std_shift_lognormal = 0.7
    # np.random.seed(seed)

    # np.random.seed(seed)
    x_values = np.linspace(x_start, x_end, n_points)
    # params = generate_uniform_sigmoid_parameters(n_tasks, slope_range, shift_range)
    params = generate_sigmoid_parameters(
        n_tasks,
        slope_range,
        shift_range,
        "lognormal",
        mean_shift_lognormal,
        std_shift_lognormal,
    )
    individual_sigmoids = get_individual(x_values, params)
    y_sigmoid_combined = y_sigmoid_combined_function(individual_sigmoids)
    mask = x_values <= g_factor_cutoff
    # fitted_params = get_sigmoid_fit_params(x_values[mask], y_sigmoid_combined[mask])
    # plot_sigmoid_functions(
    #     x_values,
    #     individual_sigmoids,
    #     y_sigmoid_combined,
    #     fitted_params,
    #     n_tasks,
    #     scale_factor,
    #     g_factor_cutoff,
    #     params,
    #     upper_quantile,
    # )
    
    fitted_params = get_lognormal_cdf_fit_params(x_values[mask], y_sigmoid_combined[mask])
    print(fitted_params)
    plot_sigmoid_functions_lognormal_cdf_fit(
        x_values,
        individual_sigmoids,
        y_sigmoid_combined,
        fitted_params,
        n_tasks,
        scale_factor,
        g_factor_cutoff,
        params,
        upper_quantile,
    )


if __name__ == "__main__":
    main()
