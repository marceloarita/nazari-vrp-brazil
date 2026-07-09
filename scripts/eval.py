"""
Evaluation script — compares NazariAgent vs OR-Tools on the same instances.

Usage:
    uv run scripts/eval.py --checkpoint checkpoints/epoch_01000.pt --n_customers 20
    uv run scripts/eval.py --checkpoint checkpoints/epoch_01000.pt --n_customers 20 --plot
"""

import argparse
import torch
from src.environment import generate_batch, VRPEnvironment
from src.agent import NazariAgent
from src.trainer import rollout
from src.solver import solve_batch
from src.utils import gap_percent
from src.plot import plot_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--n_customers", type=int, default=20)
    parser.add_argument("--n_instances", type=int, default=32)
    parser.add_argument("--vehicle_capacity", type=int, default=20)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--time_limit", type=int, default=10, help="OR-Tools time limit per instance (seconds)")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_idx", type=int, default=0)
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)

    print(f"VRP{args.n_customers} | {args.n_instances} instances | device: {device}\n")

    # --- Shared instances ---
    coords, demands = generate_batch(args.n_instances, args.n_customers, vehicle_capacity=args.vehicle_capacity, device=device)

    # --- Nazari agent ---
    nazari_agent = NazariAgent(args.checkpoint, embed_dim=args.embed_dim, device=device)
    env_nazari = VRPEnvironment(coords, demands, vehicle_capacity=args.vehicle_capacity)
    with torch.no_grad():
        _, rewards_nazari = rollout(nazari_agent, env_nazari, greedy=True)

    # --- OR-Tools ---
    print("Running OR-Tools solver...")
    ortools_tours, dist_ortools = solve_batch(coords, demands, time_limit_s=args.time_limit)

    # --- Results ---
    dist_nazari = -rewards_nazari
    gap = gap_percent(dist_nazari, dist_ortools)

    print(f"\n{'':20s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    print(f"{'OR-Tools':20s}  {dist_ortools.mean():8.4f}  {dist_ortools.std():8.4f}  {dist_ortools.min():8.4f}  {dist_ortools.max():8.4f}")
    print(f"{'Nazari (greedy)':20s}  {dist_nazari.mean():8.4f}  {dist_nazari.std():8.4f}  {dist_nazari.min():8.4f}  {dist_nazari.max():8.4f}")
    print(f"\nGap% (Nazari vs OR-Tools): {gap.mean():.2f}%  (lower is better)")

    if args.plot:
        idx = args.plot_idx
        coords = env_nazari.static[idx].cpu().numpy()
        entries = [
            {"coords": coords,
             "tour": [0] + [t[idx].item() for t in env_nazari.tour],
             "title": f"Nazari Agent (greedy)\ndist: {(-rewards_nazari[idx]):.4f}"},
            {"coords": coords,
             "tour": ortools_tours[idx],
             "title": f"OR-Tools (optimal)\ndist: {dist_ortools[idx]:.4f}"},
        ]
        import matplotlib.pyplot as plt
        plot_comparison(entries)
        plt.show()


if __name__ == "__main__":
    main()
