# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Google's reported token processing milestones
# July 2024: 980 trillion monthly tokens
# ~September 2024: 1.3 quadrillion monthly tokens (announced as "this summer" milestone)

dates = np.array([2024.5, 2024.75])  # July 2024, ~Sept 2024
tokens_trillion = np.array([980, 1300])  # in trillions

# Exponential model: y = a * exp(b * t)
def exponential(t, a, b):
    return a * np.exp(b * t)

# Fit the exponential curve
# Shift time to avoid numerical issues with large exponents
t_shifted = dates - 2024
popt, pcov = curve_fit(exponential, t_shifted, tokens_trillion, p0=[900, 1])

# Generate points for the fitted curve (extend into past and future)
t_fit = np.linspace(-2, 3, 200)  # 2 years before and after 2024
dates_fit = t_fit + 2024
tokens_fit = exponential(t_fit, *popt)

# %%
# Create the visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Plot data points
ax.scatter(dates, tokens_trillion, s=100, zorder=5, label='Reported Data', color='tab:blue')

# Plot exponential fit
ax.plot(dates_fit, tokens_fit, '--', linewidth=2, label='Exponential Fit', color='tab:orange')

# Add annotations for data points
ax.annotate('July 2024\n980T tokens', xy=(2024.5, 980), 
            xytext=(2024.3, 700), fontsize=9,
            arrowprops=dict(arrowstyle='->', alpha=0.7))
ax.annotate('Sept 2024\n1.3Q tokens', xy=(2024.75, 1300), 
            xytext=(2024.85, 1500), fontsize=9,
            arrowprops=dict(arrowstyle='->', alpha=0.7))

# Log scale for y-axis
ax.set_yscale('log')

# Styling
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Monthly Tokens Processed (Trillions)', fontsize=11)
ax.set_title('Google AI Token Processing Growth', fontsize=14, fontweight='bold')

# Format x-axis with dates (2 years on either side)
ax.set_xlim(2022, 2027)
ax.set_xticks(range(2022, 2027))
plt.xticks(rotation=45, ha='right')

ax.grid(True, alpha=0.3, which='both')
ax.legend(loc='upper left')

# Calculate and display growth rate
growth_rate = popt[1]  # b parameter
doubling_time = np.log(2) / growth_rate  # in years
monthly_growth = (np.exp(growth_rate / 12) - 1) * 100

fig.text(0.99, 0.02, f'Doubling time: {doubling_time:.2f} years | Monthly growth: ~{monthly_growth:.1f}%', 
         fontsize=9, color='gray', ha='right', va='bottom')

plt.tight_layout()
plt.savefig('token_scaling_growth.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Print statistics
print("\n📊 Google Token Processing Growth")
print("=" * 45)
print(f"July 2024: {980:,} trillion tokens/month")
print(f"Sept 2024: {1300:,} trillion tokens/month (1.3 quadrillion)")
print(f"\nGrowth rate: {growth_rate:.2f}/year")
print(f"Doubling time: {doubling_time:.2f} years ({doubling_time*12:.1f} months)")
print(f"Monthly growth rate: ~{monthly_growth:.1f}%")

# Project future values
print(f"\nProjections (if trend continues):")
print(f"  2025: {exponential(1, *popt)/1000:.2f} quadrillion tokens/month")
print(f"  2026: {exponential(2, *popt)/1000:.2f} quadrillion tokens/month")
print(f"  2027: {exponential(3, *popt)/1000:.2f} quadrillion tokens/month")
print(f"  2029: {exponential(5, *popt)/1000:.2f} quadrillion tokens/month")

