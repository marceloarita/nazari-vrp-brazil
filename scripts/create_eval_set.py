"""
Generate a fixed evaluation set and solve with OR-Tools once.
The resulting file is reused by eval.py to skip OR-Tools on every run.

Usage:
    uv run scripts/create_eval_set.py --n_customers 10 --vehicle_capacity 20 --n_instances 32
    uv run scripts/create_eval_set.py --n_customers 20 --vehicle_capacity 30 --n_instances 32

Output:
    artifacts/instances/vrp{n}_cap{cap}_n{instances}.pt
"""

import argparse
import os
import time
import torch
from src.environment import generate_batch
from src.solver import solve_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_customers", type=int, required=True)
    parser.add_argument("--vehicle_capacity", type=int, required=True)
    parser.add_argument("--n_instances", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time_limit", type=int, default=30, help="OR-Tools time limit per instance (s)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs("artifacts/instances", exist_ok=True)

    out_path = f"artifacts/instances/vrp{args.n_customers}_cap{args.vehicle_capacity}_n{args.n_instances}.pt"

    print(f"Generating {args.n_instances} instances - VRP{args.n_customers}, cap={args.vehicle_capacity}, seed={args.seed}")
    coords, demands = generate_batch(args.n_instances, args.n_customers, vehicle_capacity=args.vehicle_capacity)

    print(f"Running OR-Tools (time_limit={args.time_limit}s per instance)...")
    t0 = time.time()
    ortools_tours, dist_ortools = solve_batch(coords, demands, time_limit_s=args.time_limit)
    t_ortools = time.time() - t0

    torch.save({
        "coords": coords,
        "demands": demands,
        "ortools_tours": ortools_tours,
        "dist_ortools": dist_ortools,
        "n_customers": args.n_customers,
        "vehicle_capacity": args.vehicle_capacity,
        "n_instances": args.n_instances,
        "seed": args.seed,
        "t_ortools_s": t_ortools,
    }, out_path)

    print(f"OR-Tools done in {t_ortools:.1f}s  (mean dist: {dist_ortools.mean():.4f})")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
