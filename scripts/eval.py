"""
Evaluation script — compares NazariAgent vs OR-Tools on the same instances.

Usage (with pre-built eval set — recommended):
    uv run scripts/create_eval_set.py --n_customers 10 --vehicle_capacity 20
    uv run scripts/eval.py --checkpoint artifacts/checkpoints/vrp10_cap20_ema/epoch_10000.pt --eval_set artifacts/instances/vrp10_cap20_n32.pt

Usage (ad-hoc, generates instances + runs OR-Tools on the fly):
    uv run scripts/eval.py --checkpoint artifacts/checkpoints/epoch_01000.pt --n_customers 10

Logs are appended to artifacts/results/eval_log.csv.
"""

import argparse
import csv
import os
import time
from datetime import datetime

import torch

from src.agent import NazariAgent
from src.environment import VRPEnvironment, generate_batch
from src.plot import plot_comparison
from src.solver import solve_batch
from src.trainer import rollout
from src.utils import gap_percent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--eval_set", type=str, default=None,
                        help="Path to pre-built eval set (.pt). If given, skips OR-Tools.")
    # Ad-hoc options (used only when --eval_set is not provided)
    parser.add_argument("--n_customers", type=int, default=10)
    parser.add_argument("--n_instances", type=int, default=32)
    parser.add_argument("--vehicle_capacity", type=int, default=20)
    parser.add_argument("--time_limit", type=int, default=10, help="OR-Tools time limit (s), ad-hoc only")
    # Other
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--samples", type=int, default=1,
                        help="1 = greedy decoding; N>1 = sample N tours/instance and keep the best")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log", type=str, default="artifacts/results/eval_log.csv")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_idx", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)

    # --- Load or generate instances ---
    if args.eval_set:
        data = torch.load(args.eval_set, map_location=device, weights_only=False)
        coords = data["coords"].to(device)
        demands = data["demands"].to(device)
        dist_ortools = data["dist_ortools"].to(device)
        ortools_tours = data["ortools_tours"]
        n_customers = data["n_customers"]
        vehicle_capacity = data["vehicle_capacity"]
        n_instances = data["n_instances"]
        t_ortools = data.get("t_ortools_s", 0.0)
        print(f"Loaded eval set: VRP{n_customers}, cap={vehicle_capacity}, {n_instances} instances")
        print(f"OR-Tools (cached)  mean: {dist_ortools.mean():.4f}  solved in: {t_ortools:.1f}s total ({t_ortools/n_instances:.2f}s/instance)\n")
    else:
        n_customers = args.n_customers
        vehicle_capacity = args.vehicle_capacity
        n_instances = args.n_instances
        coords, demands = generate_batch(n_instances, n_customers, vehicle_capacity=vehicle_capacity, device=device)
        print(f"VRP{n_customers} | {n_instances} instances | device: {device}\n")
        print("Running OR-Tools solver...")
        t0 = time.time()
        ortools_tours, dist_ortools = solve_batch(coords, demands, time_limit_s=args.time_limit)
        t_ortools = time.time() - t0
        print(f"OR-Tools done in {t_ortools:.1f}s\n")

    # --- Nazari agent ---
    nazari_agent = NazariAgent(args.checkpoint, embed_dim=args.embed_dim, device=device)
    env_nazari = VRPEnvironment(coords, demands, vehicle_capacity=vehicle_capacity)

    t0 = time.time()
    with torch.no_grad():
        if args.samples <= 1:
            _, rewards_nazari = rollout(nazari_agent, env_nazari, greedy=True)
        else:
            rewards_nazari = None
            for _ in range(args.samples):
                _, r = rollout(nazari_agent, env_nazari, greedy=False)  # stochastic
                rewards_nazari = r if rewards_nazari is None else torch.maximum(rewards_nazari, r)
    t_nazari = time.time() - t0

    # --- Results ---
    dist_nazari = -rewards_nazari
    gap = gap_percent(dist_nazari, dist_ortools)
    label = "Nazari (greedy)" if args.samples <= 1 else f"Nazari (best-{args.samples})"

    print(f"{'':20s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}  {'time':>8s}")
    print(f"{'OR-Tools':20s}  {dist_ortools.mean():8.4f}  {dist_ortools.std():8.4f}  {dist_ortools.min():8.4f}  {dist_ortools.max():8.4f}  {t_ortools:7.2f}s")
    print(f"{label:20s}  {dist_nazari.mean():8.4f}  {dist_nazari.std():8.4f}  {dist_nazari.min():8.4f}  {dist_nazari.max():8.4f}  {t_nazari:7.2f}s")
    print(f"\nGap% (Nazari vs OR-Tools): {gap.mean():.2f}%  (lower is better)")

    # --- Log ---
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    log_exists = os.path.exists(args.log)
    with open(args.log, "a", newline="") as f:
        writer = csv.writer(f)
        if not log_exists:
            writer.writerow(["timestamp", "checkpoint", "n_customers", "vehicle_capacity",
                             "n_instances", "nazari_mean", "nazari_std",
                             "ortools_mean", "ortools_std", "gap_mean",
                             "t_nazari_s", "t_ortools_s"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            args.checkpoint,
            n_customers,
            vehicle_capacity,
            n_instances,
            f"{dist_nazari.mean():.4f}",
            f"{dist_nazari.std():.4f}",
            f"{dist_ortools.mean():.4f}",
            f"{dist_ortools.std():.4f}",
            f"{gap.mean():.2f}",
            f"{t_nazari:.3f}",
            f"{t_ortools:.3f}",
        ])
    print(f"\nLogged → {args.log}")

    # --- Plot ---
    if args.plot:
        idx = args.plot_idx
        coords_np = env_nazari.static[idx].cpu().numpy()
        entries = [
            {"coords": coords_np,
             "tour": [0] + [t[idx].item() for t in env_nazari.tour],
             "title": f"Nazari Agent (greedy)\ndist: {dist_nazari[idx]:.4f}"},
            {"coords": coords_np,
             "tour": ortools_tours[idx],
             "title": f"OR-Tools\ndist: {dist_ortools[idx]:.4f}"},
        ]
        import matplotlib.pyplot as plt
        plot_comparison(entries)
        plt.show()


if __name__ == "__main__":
    main()
