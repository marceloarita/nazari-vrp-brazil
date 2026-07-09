"""Quick visual test for the OR-Tools solver on a random VRP10 instance."""

import matplotlib.pyplot as plt
import torch
from src.environment import generate_batch
from src.solver import solve_batch
from src.plot import plot_comparison

torch.manual_seed(0)

N = 10
coords, demands = generate_batch(1, N, vehicle_capacity=20)

print(f"Solving VRP{N}...")
tours, distances = solve_batch(coords, demands)

print(f"Tour:     {tours[0]}")
print(f"Distance: {distances[0]:.4f}")

entries = [{"coords": coords[0], "tour": tours[0], "title": f"OR-Tools | VRP{N} | dist: {distances[0]:.4f}"}]
plot_comparison(entries)
plt.show()
