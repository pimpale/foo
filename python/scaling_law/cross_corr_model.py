# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scipy.optimize as opt

# convolve normal curve with step function, contrast this with convolving normal curve with sigmoid function

# normal curve
x = np.linspace(-10, 10, 2000)
y = stats.norm.pdf(x, 0, 1)

x_extended = np.linspace(2*x[0], 2*x[-1], 2*len(x)-1)

# step function
step = np.zeros_like(x_extended)
step[x_extended > 0] = 1

# convolve normal curve with step function
convolution = np.convolve(y, step, mode='valid')
convolution = convolution * (x[1] - x[0])

plt.plot(x, y, label='normal curve')
plt.plot(x_extended, step, label='step function')
plt.plot(x, convolution, label='convolution')
plt.xlim(-10, 10)
plt.legend()
plt.show()


# now convolve normal curve with sigmoid function
sigmoid = stats.norm.cdf(x_extended, 0, 1)

convolution = np.convolve(y, sigmoid, mode='valid')
convolution = convolution * (x[1] - x[0])

plt.plot(x, y, label='normal curve')
plt.plot(x_extended, sigmoid, label='sigmoid function')
plt.plot(x, convolution, label='convolution')
plt.xlim(-10, 10)
plt.legend()

plt.show()


# log normal distribution
y_lognormal = stats.lognorm.pdf(x, 1, 0, 1)

lognormal_convolution = np.convolve(y_lognormal, step, mode='valid')
lognormal_convolution = lognormal_convolution * (x[1] - x[0])

plt.plot(x, y_lognormal, label='log normal curve')
plt.plot(x_extended, step, label='step function')
plt.plot(x, lognormal_convolution, label='convolution')
plt.xlim(-10, 10)
plt.legend()
plt.show()

lognormal_convolution = np.convolve(y_lognormal, sigmoid, mode='valid')
lognormal_convolution = lognormal_convolution * (x[1] - x[0])

plt.plot(x, y_lognormal, label='log normal curve')
plt.plot(x_extended, sigmoid, label='sigmoid function')
plt.plot(x, lognormal_convolution, label='convolution')
plt.xlim(-10, 10)
plt.legend()
plt.show()

# fit a lognormal distribution to the convolution

def lognormal_cdf(x, loc, sigma, mu):
    return stats.lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu))

def lognormal_pdf(x, loc, sigma, mu):
    return stats.lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu))

def norm_cdf(x, loc, scale):
    return stats.norm.cdf(x, loc=loc, scale=scale)

def get_lognormal_cdf_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = opt.curve_fit(
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

def get_norm_cdf_fit_params(
    x_values: np.ndarray, y_values: np.ndarray
) -> tuple[float, float]:
    assert len(x_values) == len(y_values)
    popt, _ = opt.curve_fit(
        norm_cdf,
        x_values,
        y_values,
        p0=[
            # center at the median
            0, 1
        ],
        bounds=([-100, 0.01], [20, 10]),
        maxfev=5000
    )
    return popt

lognormal_fit_params = get_lognormal_cdf_fit_params(x, lognormal_convolution)
print("loc, sigma, mu", lognormal_fit_params)

normal_fit_params = get_norm_cdf_fit_params(x, lognormal_convolution)
print("loc, scale", normal_fit_params)

plt.plot(x, lognormal_convolution, label='convolution (True distribution)')
plt.plot(x, lognormal_cdf(x, *lognormal_fit_params), label='lognormal cdf best fit')
plt.plot(x, norm_cdf(x, *normal_fit_params), label='normal cdf best fit')
plt.xlim(-3, 5)
plt.legend()
