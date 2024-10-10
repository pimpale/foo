#%%

from scipy.stats import lognorm
from matplotlib import pyplot as plt
import numpy as np
from ipywidgets import interact
import ipywidgets as widgets



#%%

@interact(
        loc=widgets.FloatSlider(min=-10, max=10, value=0),
        sigma=widgets.FloatSlider(min=0.1, max=10, value=1),
        mu=widgets.FloatSlider(min=0.1, max=100, value=1),
)
def plot_lognormal_cdf(loc, sigma, mu):
    x = np.linspace(-10, 30, 100)
    plt.plot(x, lognorm.cdf(x, loc=loc, s=sigma, scale=np.exp(mu)))
    plt.plot(x, lognorm.pdf(x, loc=loc, s=sigma, scale=np.exp(mu)))
    plt.ylim(0, 2)
    plt.show()
    
