"""
Part II — compare the learning-signal lever: H1 (Kool baseline) vs POMO.
Both per-zone models, same held-out C5 test set, vs OR-Tools.
Reports greedy, best-of-N, and best-of-N + 2-opt for each.

Run:
    uv run experiments/part2_sp/eval_h1_vs_pomo.py --cluster 5 --samples 128
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
from src.two_opt import two_opt, tour_length

INST = Path("artifacts/instances")
CKPT = Path("artifacts/checkpoints")


def get_tour(env, i):
    return [0] + [t[i].item() for t in env.tour]


def evaluate(ckpt, coords, demands, cap, dist_ort, samples, dev):
    agent = NazariAgent(str(ckpt), device=dev)
    cn = coords.cpu().numpy()
    B = coords.shape[0]

    # greedy
    env = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    with torch.no_grad():
        _, r_g = rollout(agent, env, greedy=True)
    d_greedy = (-r_g).cpu().numpy()

    # best-of-N (track best tour per instance)
    env_s = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    best_rew = torch.full((B,), float("-inf")); best_tours = [None] * B
    with torch.no_grad():
        for _ in range(samples):
            _, r = rollout(agent, env_s, greedy=False)
            for i in range(B):
                if r[i] > best_rew[i]:
                    best_rew[i] = r[i]; best_tours[i] = get_tour(env_s, i)
    d_best = (-best_rew).cpu().numpy()
    d_best_2o = np.array([tour_length(cn[i], two_opt(cn[i], best_tours[i])) for i in range(B)])

    g = lambda d: float(np.mean((d - dist_ort) / dist_ort * 100.0))
    return g(d_greedy), g(d_best), g(d_best_2o)


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
    coords, demands = data["coords"].to(dev), data["demands"].to(dev)
    cap = float(data["vehicle_capacity"]); dist_ort = data["dist_ortools"].cpu().numpy()

    models = [
        ("H1 (Kool baseline)", CKPT / f"vrp20_real_c{c}__kool__base/epoch_20000.pt"),
        ("POMO",               CKPT / f"vrp20_real_c{c}_pomo/epoch_20000.pt"),
    ]

    print(f"Zone C{c}  |  held-out n={coords.shape[0]}  |  N(best-of)={args.samples}\n")
    print(f"{'Model':<22} {'greedy':>8} {'best-'+str(args.samples):>9} {'best+2opt':>11}")
    print("-" * 54)
    for name, ckpt in models:
        gg, gb, gb2 = evaluate(ckpt, coords, demands, cap, dist_ort, args.samples, dev)
        print(f"{name:<22} {gg:>7.1f}% {gb:>8.1f}% {gb2:>10.1f}%")


if __name__ == "__main__":
    main()
