"""
Generate comparison plots: Nazari (greedy) vs OR-Tools for N instances.

Usage:
    uv run scripts/plot_eval.py --checkpoint checkpoints/vrp10_cap20_ema/epoch_10000.pt \
                                --eval_set eval_sets/vrp10_cap20_n32.pt \
                                --n_plots 10 \
                                --out_dir plots/vrp10_ema_vs_ortools
"""

import argparse
import os
import time

import matplotlib.pyplot as plt
import torch

from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.plot import plot_comparison
from src.trainer import rollout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_set", type=str, required=True)
    parser.add_argument("--n_plots", type=int, default=10)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_dir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load eval set
    data = torch.load(args.eval_set, map_location=args.device, weights_only=False)
    coords = data["coords"].to(args.device)
    demands = data["demands"].to(args.device)
    dist_ortools = data["dist_ortools"].to(args.device)
    ortools_tours = data["ortools_tours"]
    n_customers = data["n_customers"]
    vehicle_capacity = data["vehicle_capacity"]
    n_instances = data["n_instances"]
    t_ortools_total = data.get("t_ortools_s", 0.0)
    t_ortools_per = t_ortools_total / n_instances  # avg per instance

    # Run Nazari (timed)
    agent = NazariAgent(args.checkpoint, embed_dim=args.embed_dim, device=args.device)
    env = VRPEnvironment(coords, demands, vehicle_capacity=vehicle_capacity)
    t0 = time.time()
    with torch.no_grad():
        _, rewards = rollout(agent, env, greedy=True)
    t_nazari_per = (time.time() - t0) / n_instances  # avg per instance
    dist_nazari = -rewards

    n_plots = min(args.n_plots, coords.size(0))
    label = os.path.basename(os.path.dirname(args.checkpoint))

    print(f"Saving {n_plots} plots to {args.out_dir}/")
    for i in range(n_plots):
        coords_np = env.static[i].cpu().numpy()
        nazari_tour = [0] + [t[i].item() for t in env.tour]

        fig = plot_comparison([
            {
                "coords": coords_np,
                "tour": nazari_tour,
                "title": (
                    f"Nazari ({label})\n"
                    f"dist: {dist_nazari[i]:.3f}  |  time: {t_nazari_per*1000:.1f}ms/inst"
                ),
            },
            {
                "coords": coords_np,
                "tour": ortools_tours[i],
                "title": (
                    f"OR-Tools\n"
                    f"dist: {dist_ortools[i]:.3f}  |  time: {t_ortools_per:.1f}s/inst"
                ),
            },
        ])

        gap = (dist_nazari[i].item() - dist_ortools[i].item()) / dist_ortools[i].item() * 100
        fig.suptitle(
            f"VRP{n_customers} — Instance {i+1}  |  Gap: {gap:+.1f}%",
            fontsize=13, y=1.02, color="#2d3436"
        )

        path = os.path.join(args.out_dir, f"instance_{i+1:02d}.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{i+1}/{n_plots}] gap={gap:+.1f}%  -> {path}")

    print("Done.")


if __name__ == "__main__":
    main()
