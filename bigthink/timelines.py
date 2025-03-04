# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

unnormalized_probs = [
    (2025.0, 2026.0, 0),
    (2026.0, 2027.0, 1),
    (2027.0, 2028.0, 4),
    (2028.0, 2029.0, 10),
    (2029.0, 2030.0, 10),
    (2030.0, 2035.0, 20),
    (2035.0, 2050.0, 50),
]

def normalize_probs(unnormalized_probs):
    total = sum([p for _, _, p in unnormalized_probs])
    return [(start, end, p / total) for start, end, p in unnormalized_probs]


def plot_prob(unnormalized_probs):
    # Normalize probabilities
    norm_probs = normalize_probs(unnormalized_probs)
    
    # Calculate probability density (probability per year)
    prob_densities = [p / (end - start) for start, end, p in norm_probs]
    
    # Get x values (years)
    x_values = [start for start, _, _ in norm_probs]
    # Add the final end year
    x_values.append(norm_probs[-1][1])
    
    # Plot using stairs with probability densities
    plt.stairs(
        values=prob_densities,
        edges=x_values,
        label='Probability Density (per year)',
        fill=True,
        alpha=0.3
    )
    
    plt.grid(True)
    plt.xlabel('Year')
    plt.ylabel('Probability Density')
    plt.title('Timeline Probability Density')
    plt.legend()
    plt.show()

def plot_cumulative_prob(unnormalized_probs):
    # Normalize probabilities
    norm_probs = normalize_probs(unnormalized_probs)
    
    # Calculate cumulative probabilities
    cum_prob = np.cumsum([p for _, _, p in norm_probs])
    
    # Get x values (years)
    x_values = [start for start, _, _ in norm_probs]
    # Add the final end year
    x_values.append(norm_probs[-1][1])
    
    # Plot using stairs - cum_prob is values, x_values is edges
    plt.stairs(
        values=cum_prob,
        edges=x_values,
        label='Cumulative Probability',
        fill=True,
        alpha=0.3
    )
    
    plt.grid(True)
    plt.xlabel('Year')
    plt.ylabel('Cumulative Probability')
    plt.title('Timeline Cumulative Probability')
    plt.legend()
    plt.show()
    
    
plot_prob(unnormalized_probs)
    
    
plot_cumulative_prob(unnormalized_probs)

print(np.round([x for _, _, x in normalize_probs(unnormalized_probs)], 2))