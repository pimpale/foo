import numpy as np
import matplotlib.pyplot as plt


grass = 10000
chickens = 10
foxes = 1


grass_birth_rate = 0.9
grass_death_rate = 0.015
chicken_birth_rate = 0.0015
chicken_death_rate = 0.5
fox_birth_rate = 0.015
fox_death_rate = 0.4

delta_time = 0.001
cycles = 4500

for t in range(0, cycles):
    updated_grass = grass + delta_time * (grass_birth_rate * grass  - grass_death_rate * chickens * grass)
    updated_chickens = chickens + delta_time * (chicken_birth_rate * chickens * grass - chicken_death_rate * foxes * chickens)
    updated_foxes = foxes + delta_time * (-fox_death_rate * foxes + fox_birth_rate * foxes * chickens)

    grass = max(updated_grass, 0)
    chickens = max(updated_chickens, 0)
    foxes = max(updated_foxes, 0)

    plt.scatter(t, grass, c="green")
    plt.scatter(t, chickens, c="yellow")
    plt.scatter(t, foxes, c="red")
    if t%110 is 0:
        plt.pause(0.001)


