"""
Part II / Hypothesis A4 — 2-opt post-processing on the per-zone (H1) model.

No retraining: take H1's tours (greedy and best-of-N) and run intra-trip 2-opt,
then compare gaps against OR-Tools on the held-out test set.

Run:
    uv run experiments/part2_sp/eval_2opt.py --cluster 5 --samples 128
"""

import argparse
from pathlib import Path

import torch
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.trainer import rollout
from src.utils import gap_percent
from src.two_opt import two_opt, tour_length

INST = Path("artifacts/instances")


def get_tour(env, i):
    return [0] + [t[i].item() for t in env.tour]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", type=int, default=5)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    torch.manual_seed(args.seed)
    c, dev = args.cluster, args.device

    data = torch.load(next(INST.glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt")),
                      map_location=dev, weights_only=False)
    coords  = data["coords"].to(dev)
    demands = data["demands"].to(dev)
    cap     = float(data["vehicle_capacity"])
    dist_ort = data["dist_ortools"].to(dev)
    B = coords.shape[0]
    coords_np = coords.cpu().numpy()

    ckpt = f"artifacts/checkpoints/vrp20_real_c{c}__kool__base/epoch_20000.pt"
    agent = NazariAgent(ckpt, device=dev)

    # --- Greedy tours ---
    env = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    with torch.no_grad():
        _, r_g = rollout(agent, env, greedy=True)
    dist_greedy = (-r_g).cpu().numpy()
    greedy_tours = [get_tour(env, i) for i in range(B)]

    # --- Best-of-N tours (track best tour per instance) ---
    env_s = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    best_rew = torch.full((B,), float("-inf"))
    best_tours = [None] * B
    with torch.no_grad():
        for _ in range(args.samples):
            _, r = rollout(agent, env_s, greedy=False)
            for i in range(B):
                if r[i] > best_rew[i]:
                    best_rew[i] = r[i]; best_tours[i] = get_tour(env_s, i)
    dist_best = (-best_rew).cpu().numpy()

    # --- Apply 2-opt ---
    def opt(tours):
        return np.array([tour_length(coords_np[i], two_opt(coords_np[i], tours[i])) for i in range(B)])

    dist_greedy_2o = opt(greedy_tours)
    dist_best_2o   = opt(best_tours)

    ort = dist_ort.cpu().numpy()
    def gap(d):
        return float(np.mean((d - ort) / ort * 100.0))

    print(f"Zone C{c}  |  held-out n={B}  |  N(best-of)={args.samples}  |  H1 model\n")
    print(f"{'Decoding':<22} {'gap':>8} {'+2-opt':>8}")
    print("-" * 40)
    print(f"{'greedy':<22} {gap(dist_greedy):>7.1f}% {gap(dist_greedy_2o):>7.1f}%")
    print(f"{'best-of-'+str(args.samples):<22} {gap(dist_best):>7.1f}% {gap(dist_best_2o):>7.1f}%")


if __name__ == "__main__":
    main()
