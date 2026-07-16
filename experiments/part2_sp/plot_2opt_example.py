"""
Part II — real before/after example of 2-opt on the running best (best-of-128), zone C5.
Picks the held-out instance where 2-opt improves the best-of-128 tour the most,
and plots it before vs after, with the gap vs OR-Tools.

Output: docs/images/two_opt_example_c5.png
Run:    uv run experiments/part2_sp/plot_2opt_example.py
"""
import argparse
from pathlib import Path
import torch, numpy as np, matplotlib.pyplot as plt
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.trainer import rollout
from src.two_opt import two_opt, tour_length
from src.plot import plot_route

p = argparse.ArgumentParser()
p.add_argument("--cluster", type=int, default=5)
p.add_argument("--samples", type=int, default=128)
p.add_argument("--seed", type=int, default=42)
a = p.parse_args()
torch.manual_seed(a.seed); c, dev = a.cluster, "cpu"

data = torch.load(next(Path("artifacts/instances").glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt")),
                  map_location=dev, weights_only=False)
coords, demands = data["coords"], data["demands"]
cap = float(data["vehicle_capacity"]); dist_ort = data["dist_ortools"].numpy()
B = coords.shape[0]; raw_dem = (demands * 30).round().numpy()
agent = NazariAgent(f"artifacts/checkpoints/vrp20_real_c{c}__kool__base/epoch_20000.pt", device=dev)

# best-of-N per instance
env = VRPEnvironment(coords, demands, vehicle_capacity=cap)
best_rew = torch.full((B,), float("-inf")); best_tours = [None]*B
with torch.no_grad():
    for _ in range(a.samples):
        _, r = rollout(agent, env, greedy=False)
        for i in range(B):
            if r[i] > best_rew[i]:
                best_rew[i] = r[i]; best_tours[i] = [0]+[t[i].item() for t in env.tour]
cn = coords.numpy()
d_before = (-best_rew).numpy()
opt_tours = [two_opt(cn[i], best_tours[i]) for i in range(B)]
d_after = np.array([tour_length(cn[i], opt_tours[i]) for i in range(B)])

idx = int(np.argmax(d_before - d_after))   # biggest 2-opt gain
gb = (d_before[idx]-dist_ort[idx])/dist_ort[idx]*100
ga = (d_after[idx]-dist_ort[idx])/dist_ort[idx]*100
print(f"instance {idx}: before {d_before[idx]:.3f} ({gb:.1f}%) -> after {d_after[idx]:.3f} ({ga:.1f}%)")

fig, ax = plt.subplots(1, 2, figsize=(11, 5.6))
xy = cn[idx]; xmin,ymin = xy.min(0); xmax,ymax = xy.max(0)
cx,cy = (xmin+xmax)/2,(ymin+ymax)/2; half = max(xmax-xmin, ymax-ymin)/2*1.12
plot_route(xy, best_tours[idx], ax=ax[0], demands=raw_dem[idx],
           title=f"best-of-128   dist = {d_before[idx]:.3f}   gap = {gb:.1f}%")
plot_route(xy, opt_tours[idx], ax=ax[1], demands=raw_dem[idx],
           title=f"best-of-128 + 2-opt   dist = {d_after[idx]:.3f}   gap = {ga:.1f}%")
for x in ax:
    x.set_xlim(cx-half,cx+half); x.set_ylim(cy-half,cy+half); x.set_aspect("equal")
    x.set_xticks([]); x.set_yticks([])
plt.suptitle("2-opt on a real C5 instance — uncrossing the best-of-128 tour", fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig("docs/images/two_opt_example_c5.png", dpi=110, bbox_inches="tight"); plt.close()
print("saved docs/images/two_opt_example_c5.png")
