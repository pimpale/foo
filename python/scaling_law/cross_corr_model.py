# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

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
sigmoid = 1/(1+np.exp(-x_extended))

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
