import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from scipy.optimize import curve_fit
from scipy.special import erf
from scipy.stats import lognorm
from typing import Literal, Union


def sigmoid(x: np.ndarray, slope: float, shift: float) -> np.ndarray:
    return 1 / (1 + np.exp(-slope * (x - shift)))

def lognormal_cdf(x: np.ndarray, shift: float, s: float, scale: float) -> np.ndarray:
    return lognorm.cdf(x, loc=shift, scale=scale)

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
            np.random.lognormal(mean_shift, std_shift, n_tasks), *shift_range
        )
    else:
        raise ValueError("Distribution must be 'uniform', 'gaussian', or 'lognormal'")
    return list(zip(k_values, x0_values, strict=True))


def y_sigmoid_combined_function(individual_sigmoids: list[np.ndarray]) -> np.ndarray:
    return np.mean(individual_sigmoids, axis=0)


def get_sigmoid_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float]:
    popt, _ = curve_fit(
        sigmoid,
        x_values,
        y_values,
        p0=[1, np.median(x_values)],
    )
    return popt

def get_lognormal_cdf_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float, float]:
    popt, _ = curve_fit(
        lognormal_cdf,
        x_values,
        y_values,
        p0=[
            # center at the median
            np.median(x_values), 1, 1
        ],
    )
    return popt

def interpolate_midpoint(
    x0: float, y0: float, x1: float, y1: float, y_between: float
) -> float:
    return x0 + (y_between - y0) * (x1 - x0) / (y1 - y0)


def find_x_quantile(
    x_values: np.ndarray, sigmoid_values: np.ndarray, y_target: float
) -> float:
    i = np.searchsorted(sigmoid_values, y_target)
    if i >= len(x_values):
        return x_values[-1]
    elif i == 0:
        return x_values[0]
    else:
        return interpolate_midpoint(
            x0=x_values[i - 1],
            y0=sigmoid_values[i - 1],
            x1=x_values[i],
            y1=sigmoid_values[i],
            y_between=y_target,
        )


def get_individual(
    x_values: np.ndarray, params: list[tuple[float, float]]
) -> list[np.ndarray]:
    return [sigmoid(x_values, slope, shift) for slope, shift in params]


def get_y_sigmoid_combined(
    x_values: np.ndarray, params: list[tuple[float, float]]
) -> np.ndarray:
    individual_sigmoids = get_individual(x_values, params)
    y_sigmoid_combined = y_sigmoid_combined_function(individual_sigmoids)
    return y_sigmoid_combined


def make_prediction(
    x_values: np.ndarray,
    y_sigmoid_combined: np.ndarray,
    g_factor_cutoff: float,
    upper_quantile: float,
) -> float:
    mask = x_values <= g_factor_cutoff
    params = get_sigmoid_fit_params(x_values[mask], y_sigmoid_combined[mask])
    y_sigmoid_curve = sigmoid(x_values, *params)
    predicted_g = find_x_quantile(x_values, y_sigmoid_curve, upper_quantile)
    return predicted_g


# def get_target_g(
#     x_values: np.ndarray,
#     slope_range: tuple[float, float],
#     shift_range: tuple[float, float],
#     upper_quantile: float,
# ) -> float:
#     n_tasks = 2000
#     shift_range = (15, 20)

#     params = generate_sigmoid_parameters(n_tasks, slope_range, shift_range, "normal")
#     y_sigmoid_combined = get_y_sigmoid_combined(x_values, params)
#     target_g = find_x_quantile(x_values, y_sigmoid_combined, upper_quantile)
#     return target_g


def show_avg_mse(
    n_tasks_list: list[int],
    repeats: int,
    slope_range: tuple[float, float],
    shift_range: tuple[float, float],
    distribution: Literal["uniform", "normal", "lognormal"],
    g_factor_cutoff: float,
    upper_quantile: float,
    x_values: np.ndarray,
    mean_shift: float = 0,
    std_shift: float = 1,
) -> list[float]:
    mse_results: list[float] = []

    for n_tasks in n_tasks_list:
        mse_sum = 0
        for _ in range(repeats):
            params = generate_sigmoid_parameters(
                n_tasks, slope_range, shift_range, distribution, mean_shift, std_shift
            )
            y_sigmoid_combined = get_y_sigmoid_combined(x_values, params)
            predicted_g = make_prediction(
                x_values, y_sigmoid_combined, g_factor_cutoff, upper_quantile
            )
            actual_g = find_x_quantile(x_values, y_sigmoid_combined, upper_quantile)
            mse_sum += (predicted_g - actual_g) ** 2

        average_mse = mse_sum / repeats
        mse_results.append(average_mse)

    # Print results
    print(f"{distribution.capitalize()} Distribution Results:")
    print("Number of Tasks | Average MSE")
    print("--------------------------------")
    for n_tasks, mse in zip(n_tasks_list, mse_results):
        print(f"{n_tasks:14d} | {mse:.6f}")

    return mse_results


def plot_combined_results(
    n_tasks_list: list[int],
    gaussian_results: list[float],
    uniform_results: list[float],
    lognormal_results: list[float],
) -> None:
    plt.figure(figsize=(12, 8))
    plt.plot(n_tasks_list, gaussian_results, marker="o", label="Normal", linewidth=2)
    plt.plot(n_tasks_list, uniform_results, marker="s", label="Uniform", linewidth=2)
    plt.plot(
        n_tasks_list, lognormal_results, marker="^", label="Lognormal", linewidth=2
    )
    plt.xlabel("Number of Tasks", fontsize=12)
    plt.ylabel("Average MSE", fontsize=12)
    plt.title("Average MSE vs Number of Tasks", fontsize=14)
    plt.legend(fontsize=10)
    plt.xscale("log")
    plt.yscale("log")

    # Set more x-ticks
    x_ticks = [10, 20, 40, 80, 160, 320, 640]
    plt.xticks(x_ticks, [str(x) for x in x_ticks])

    # Set more y-ticks
    y_ticks = [0.5, 1, 2, 4, 8, 16, 32]
    plt.yticks(y_ticks, [f"{y:.6f}" for y in y_ticks])

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:g}"))
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.6f}"))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
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
    np.random.seed(seed)

    x_values = np.linspace(x_start, x_end, n_points)

    n_tasks_list = [10, 20, 40, 80, 160, 320, 640]
    repeats = 100

    # Gaussian distribution experiment
    print("Gaussian Distribution Experiment")
    gaussian_results = show_avg_mse(
        n_tasks_list,
        repeats,
        slope_range,
        shift_range,
        "normal",
        g_factor_cutoff,
        upper_quantile,
        x_values,
        mean_shift,
        std_shift,
    )

    # Uniform distribution experiment
    print("\nUniform Distribution Experiment")
    uniform_results = show_avg_mse(
        n_tasks_list,
        repeats,
        slope_range,
        shift_range,
        "uniform",
        g_factor_cutoff,
        upper_quantile,
        x_values,
    )

    # Lognormal distribution experiment
    print("\nLognormal Distribution Experiment")
    lognormal_results = show_avg_mse(
        n_tasks_list,
        repeats,
        slope_range,
        shift_range,
        "lognormal",
        g_factor_cutoff,
        upper_quantile,
        x_values,
        mean_shift_lognormal,
        std_shift_lognormal,
    )

    # Plot combined results
    plot_combined_results(
        n_tasks_list, gaussian_results, uniform_results, lognormal_results
    )
