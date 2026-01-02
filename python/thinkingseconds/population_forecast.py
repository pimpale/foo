# %%
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('population.csv')

# Data already has world population
df['Population_Billions'] = df['all years'] / 1e9

# %%
# Create the visualization
fig, ax = plt.subplots(figsize=(12, 7))

# Split into historical and projection data (projections typically from 2024 onwards)
historical = df[df['Year'] < 2024]
projection = df[df['Year'] >= 2024]

# Plot historical data
ax.plot(historical['Year'], historical['Population_Billions'], 
        linewidth=2, label='Historical')

# Plot projection data with different style
ax.plot(projection['Year'], projection['Population_Billions'], 
        linewidth=2, linestyle='--', label='Projection (UN Medium Scenario)')

# Add a vertical line at 2024 to mark the projection start
ax.axvline(x=2024, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Highlight peak
peak_row = df.loc[df['Population_Billions'].idxmax()]
ax.scatter([peak_row['Year']], [peak_row['Population_Billions']], 
           s=80, zorder=5, color='tab:red')
ax.annotate(f'Peak: {peak_row["Population_Billions"]:.2f}B ({int(peak_row["Year"])})', 
            xy=(peak_row['Year'], peak_row['Population_Billions']),
            xytext=(peak_row['Year'] - 20, peak_row['Population_Billions'] + 0.3),
            fontsize=10,
            arrowprops=dict(arrowstyle='->', alpha=0.7))

# Styling
ax.set_xlabel('Year', fontsize=11)
ax.set_ylabel('Population (Billions)', fontsize=11)
ax.set_title('World Population Forecast (1950-2100)', fontsize=14, fontweight='bold')

ax.grid(True, alpha=0.3)
ax.set_xlim(1945, 2105)
ax.legend(loc='upper left')

# Add source citation
fig.text(0.99, 0.01, "Source: UN World Population Prospects (2024)", 
         fontsize=8, color='gray', ha='right', va='bottom')

plt.tight_layout()
plt.savefig('world_population_forecast.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
# Print some key statistics
print("\n📊 World Population Statistics")
print("=" * 40)
print(f"1950 Population: {df[df['Year'] == 1950]['Population_Billions'].values[0]:.2f} billion")
print(f"2024 Population: {df[df['Year'] == 2024]['Population_Billions'].values[0]:.2f} billion")
print(f"Peak Year: {int(peak_row['Year'])} ({peak_row['Population_Billions']:.2f} billion)")
print(f"2100 Population: {df[df['Year'] == 2100]['Population_Billions'].values[0]:.2f} billion")
