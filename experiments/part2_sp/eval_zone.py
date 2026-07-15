"""
Part II — consolidated per-zone evaluation on the held-out test set.

Reports three approaches against OR-Tools, all on the same held-out instances:
    Zero-Shot   — Part I uniform model, greedy
    Greedy      — zone-trained model, greedy
    Best-of-N   — zone-trained model, sample N tours/instance, keep the best

Then dumps per-instance distances for Best-of-N vs OR-Tools.

Run:
    uv run experiments/part2_sp/eval_zone.py --cluster 5 --samples 128
"""

import argparse
import csv
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.trainer import rollout
from src.utils import gap_percent

INST_DIR = Path("artifacts/instances")
RES_DIR  = Path("artifacts/results")
RES_DIR.mkdir(parents=True, exist_ok=True)


def run_model(ckpt, coords, demands, cap, embed_dim, device, samples):
    agent = NazariAgent(ckpt, embed_dim=embed_dim, device=device)
    env = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    with torch.no_grad():
        if samples <= 1:
            _, rewards = rollout(agent, env, greedy=True)
        else:
            rewards = None
            for _ in range(samples):
                _, r = rollout(agent, env, greedy=False)
                rewards = r if rewards is None else torch.maximum(rewards, r)
    return -rewards  # distances (B,)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster",       type=int, default=5)
    p.add_argument("--samples",       type=int, default=128)
    p.add_argument("--zero_shot_ckpt", type=str,
                   default="artifacts/checkpoints/vrp20_cap30_kool/epoch_20000.pt")
    p.add_argument("--zone_ckpt",     type=str, default=None,
                   help="default: artifacts/checkpoints/vrp20_real_c{cluster}__kool__base/epoch_20000.pt")
    p.add_argument("--embed_dim",     type=int, default=128)
    p.add_argument("--device",        type=str, default="cpu")
    p.add_argument("--seed",          type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    c = args.cluster
    zone_ckpt = args.zone_ckpt or f"artifacts/checkpoints/vrp20_real_c{c}__kool__base/epoch_20000.pt"

    # Load held-out eval set + cached OR-Tools
    path = next(INST_DIR.glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt"))
    data = torch.load(path, map_location=args.device, weights_only=False)
    coords, demands = data["coords"].to(args.device), data["demands"].to(args.device)
    cap = float(data.get("vehicle_capacity", 1.0))
    dist_ort = data.get("dist_ortools")
    if dist_ort is None:
        raise SystemExit("OR-Tools not cached in eval set. Run scripts/eval_sp.py --split heldout first.")
    dist_ort = dist_ort.to(args.device)

    print(f"Zone C{c}  |  held-out n={coords.shape[0]}  |  N(best-of)={args.samples}\n")

    # Three approaches
    d_zero = run_model(args.zero_shot_ckpt, coords, demands, cap, args.embed_dim, args.device, samples=1)
    d_grd  = run_model(zone_ckpt,           coords, demands, cap, args.embed_dim, args.device, samples=1)
    d_best = run_model(zone_ckpt,           coords, demands, cap, args.embed_dim, args.device, samples=args.samples)

    rows = [
        ("Zero-Shot (uniform)", d_zero),
        ("Greedy (zone)",       d_grd),
        (f"Best-of-{args.samples} (zone)", d_best),
    ]
    print(f"{'Approach':<24} {'Model':>8} {'OR-Tools':>10} {'Gap%':>8}")
    print("-" * 52)
    for name, d in rows:
        g = gap_percent(d, dist_ort).mean().item()
        print(f"{name:<24} {d.mean():>8.4f} {dist_ort.mean():>10.4f} {g:>7.1f}%")

    # Per-instance: Best-of-N vs OR-Tools
    out = RES_DIR / f"part2_c{c}_bestof{args.samples}_vs_ortools.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["instance", "best_of_n", "ortools", "gap_pct"])
        for i in range(len(d_best)):
            gi = (d_best[i].item() - dist_ort[i].item()) / dist_ort[i].item() * 100
            w.writerow([i, f"{d_best[i]:.4f}", f"{dist_ort[i]:.4f}", f"{gi:.2f}"])
    print(f"\nPer-instance Best-of-{args.samples} vs OR-Tools -> {out}")


if __name__ == "__main__":
    main()
