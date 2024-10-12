#%%

task_lengths = [2,
11,
35,
105,
244,
42,
65,
9,
7,
4,
120,
75,
6,
20,
368,
9,
29,
133,
159,
5,
84,
45,
51,
9,
454,
54,
204,
356,
330,
78,
52,
43,
42,
32,
6,
123,
132,
85,
11,
60,
30,
30,
30,
30,
30,
30,
30,
30,
30,
30,
30,
30,
30,
7,
7,
0,
7,
7,
7,
0,
0,
0,
7,
0,
7,
0,
7,
7,
7,
7,
7,
7,
7,
0,
7,
2,
7,
0,
0,
7,
7,
7,
7,
7,
7,
7,
7,
0,
0,
1,
15,
1,
12,
8,
8,
2,
5,
5,
60,
1,
5,
30,
1,
5,
15,
2,
5,
5,
5,
5,
5,
5,
1,
5,
5,
10,
5,
10,
0,
0,
0,
0,
0,
0,
15,
0,
300,
7,
360,
20,
10,
30,
210,
15,
15,
15,
0,
480,
0,
0,
0,
480,
5,
5,
5,
480,
60,
5,
0,
0,
0,
75,
300,
0,
0,
0,
0,
7,
37,
7,
37,
37,
37,
7,
]

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

#%%


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

task_lengths_nz = [length for length in task_lengths if length > 0]

# Plot histogram of task lengths
plt.hist(task_lengths_nz, bins=20, density=True, color="skyblue", edgecolor="black", label="Task Lengths")
# Fit a lognormal distribution to the task lengths
lognormal_params = lognorm.fit(task_lengths_nz, floc=0)

# Plot the fitted lognormal distribution
x = np.linspace(-10, 500, 512)
plt.plot(x, lognorm.pdf(x, *lognormal_params), "r-", lw=2, label="Fitted Lognormal Distribution")

plt.xlabel("Task Length")
plt.ylabel("Density")
plt.legend()
plt.show()

# plot the cdf
plt.plot(x, lognorm.cdf(x, *lognormal_params), "r-", lw=2)
plt.show()

print("Fitted lognormal parameters:", lognormal_params)


# plot histogram of log task lengths
plt.hist(np.log(task_lengths_nz), bins=20, density=True, color="skyblue", edgecolor="black", label="Log Task Lengths")
# Fit a normal distribution to the log task lengths
normal_params = np.mean(np.log(task_lengths_nz)), np.std(np.log(task_lengths_nz))

# Plot the fitted normal distribution
x = np.linspace(-10, 10, 512)
plt.plot(x, np.exp(-0.5 * ((x - normal_params[0]) / normal_params[1])**2) / (normal_params[1] * np.sqrt(2 * np.pi)), "r-", lw=2, label="Fitted Normal Distribution")

plt.xlabel("Log Task Length")
plt.ylabel("Density")
plt.legend()
plt.show()
