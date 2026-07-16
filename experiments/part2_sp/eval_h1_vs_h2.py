"""
Part II — compare per-zone models on the held-out test set:
    Zero-shot (uniform)  |  H1 (per-zone)  |  H2 (per-zone + density)   vs OR-Tools

The density model (H2) is static_dim=3, so it is fed `coords_dens` (x, y, density);
the 2D models are fed `coords`. NazariAgent auto-detects static_dim from the checkpoint.

Run:
    uv run experiments/part2_sp/eval_h1_vs_h2.py --cluster 5 --samples 128
"""

import argparse
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.trainer import rollout
from src.utils import gap_percent

INST = Path("artifacts/instances")
CKPT = Path("artifacts/checkpoints")


def run(ckpt, static, demands, cap, device, samples):
    agent = NazariAgent(str(ckpt), device=device)   # static_dim auto-detected
    env = VRPEnvironment(static, demands, vehicle_capacity=cap)
    with torch.no_grad():
        if samples <= 1:
            _, r = rollout(agent, env, greedy=True)
        else:
            r = None
            for _ in range(samples):
                _, rr = rollout(agent, env, greedy=False)
                r = rr if r is None else torch.maximum(r, rr)
    return -r


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
    coords      = data["coords"].to(dev)          # (B, N, 2)
    coords_dens = data["coords_dens"].to(dev)     # (B, N, 3)
    demands     = data["demands"].to(dev)
    cap         = float(data["vehicle_capacity"])
    dist_ort    = data["dist_ortools"].to(dev)

    models = [
        ("Zero-shot (uniform)",  CKPT / "vrp20_cap30_kool/epoch_20000.pt",          coords),
        ("H1 (per-zone)",        CKPT / "vrp20_real_c5__kool__base/epoch_20000.pt", coords),
        ("H2 (per-zone+density)", CKPT / "vrp20_real_c5_kool_dens/epoch_20000.pt",  coords_dens),
    ]

    print(f"Zone C{c}  |  held-out n={coords.shape[0]}  |  N(best-of)={args.samples}")
    print(f"OR-Tools mean dist: {dist_ort.mean():.4f}\n")
    print(f"{'Model':<24} {'Greedy gap':>12} {'Best-128 gap':>14}")
    print("-" * 52)
    for name, ckpt, static in models:
        g = gap_percent(run(ckpt, static, demands, cap, dev, 1),          dist_ort).mean().item()
        b = gap_percent(run(ckpt, static, demands, cap, dev, args.samples), dist_ort).mean().item()
        print(f"{name:<24} {g:>11.1f}% {b:>13.1f}%")


if __name__ == "__main__":
    main()
