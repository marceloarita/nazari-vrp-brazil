"""
Zero-shot evaluation on real Olist SP instances.

Runs the trained VRP20 model (no retraining) on real customer locations and
compares against OR-Tools GLS. Reports gap% per cluster and overall.

Usage:
    uv run olist_eval/run.py --checkpoint checkpoints/vrp20_kool/epoch_20000.pt

Parameters (edit at top of file or pass as CLI flags):
    --checkpoint        path to model checkpoint (.pt)
    --instance_dir      folder with Olist instance files (default: eval_sets/)
    --clusters          cluster IDs to evaluate, e.g. 1 2 3 4 5 (default: all)
    --time_limit        OR-Tools time limit in seconds per instance (default: 10)
    --embed_dim         model embed_dim, must match checkpoint (default: 128)
    --device            cpu or cuda (default: cpu)
    --seed              random seed (default: 42)
    --log               output CSV path (default: olist_eval/results.csv)
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import torch

from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.solver import solve_batch
from src.trainer import rollout
from src.utils import gap_percent


def load_instance(path: Path, device: str):
    data = torch.load(path, map_location=device, weights_only=False)
    coords  = data["coords"].to(device)
    demands = data["demands"].to(device)
    return coords, demands, data


def eval_cluster(coords, demands, vehicle_capacity, agent, time_limit, device,
                 cached_ortools=None, instance_path=None):
    """Run model + OR-Tools on a batch of instances. Returns (dist_model, dist_ortools).

    If cached_ortools is provided, OR-Tools is skipped and cached results are used.
    If instance_path is provided, newly computed OR-Tools results are saved back to the .pt file.
    """

    # OR-Tools (skip if cached)
    if cached_ortools is not None:
        dist_ortools = cached_ortools.to(device)
        t_ort = 0.0
        print("    (OR-Tools: using cached results)")
    else:
        t0 = time.time()
        _, dist_ortools = solve_batch(coords, demands,
                                      vehicle_capacity=vehicle_capacity,
                                      time_limit_s=time_limit)
        t_ort = time.time() - t0
        # Cache results back into the .pt file for future runs
        if instance_path is not None:
            data = torch.load(instance_path, map_location="cpu", weights_only=False)
            data["dist_ortools"] = dist_ortools.cpu()
            data["t_ortools_s"]  = t_ort
            torch.save(data, instance_path)

    # Model (greedy)
    env = VRPEnvironment(coords, demands, vehicle_capacity=vehicle_capacity)
    t0 = time.time()
    with torch.no_grad():
        _, rewards = rollout(agent, env, greedy=True)
    t_mdl = time.time() - t0

    dist_model = -rewards
    return dist_model, dist_ortools, t_mdl, t_ort


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",    type=str,   required=True)
    parser.add_argument("--instance_dir",  type=str,   default="eval_sets")
    parser.add_argument("--clusters",      type=int,   nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--time_limit",    type=int,   default=10,
                        help="OR-Tools seconds per instance")
    parser.add_argument("--embed_dim",     type=int,   default=128)
    parser.add_argument("--device",        type=str,   default="cpu")
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--log",           type=str,   default="olist_eval/results.csv")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device       = args.device
    instance_dir = Path(args.instance_dir)
    log_path     = Path(args.log)
    log_path.parent.mkdir(exist_ok=True)

    # Load model
    agent = NazariAgent(args.checkpoint, embed_dim=args.embed_dim, device=device)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Device     : {device}")
    print(f"OR-Tools   : {args.time_limit}s/instance")
    print(f"Clusters   : {args.clusters}\n")

    rows = []
    all_model  = []
    all_ortools = []

    header = f"{'Cluster':<10} {'n':>4} {'Model':>8} {'OR-Tools':>10} {'Gap%':>7} {'t_model':>8} {'t_ort':>8}"
    print(header)
    print("-" * len(header))

    for cid in args.clusters:
        # Find instance file — pattern: olist_sp_c{cid}_vrp20_n*_seed*.pt
        matches = sorted(instance_dir.glob(f"olist_sp_c{cid}_vrp20_n*_seed*.pt"))
        if not matches:
            print(f"  C{cid}: no instance file found in {instance_dir}/ — skipping")
            continue

        path = matches[0]
        coords, demands, meta = load_instance(path, device)
        vehicle_capacity = float(meta.get("vehicle_capacity", 1.0))
        n_instances      = coords.shape[0]

        cached = meta.get("dist_ortools")
        dist_model, dist_ortools, t_mdl, t_ort = eval_cluster(
            coords, demands, vehicle_capacity, agent, args.time_limit, device,
            cached_ortools=cached, instance_path=path,
        )

        gap = gap_percent(dist_model, dist_ortools)

        print(f"  C{cid:<8}  {n_instances:>4}  "
              f"{dist_model.mean():.4f}    {dist_ortools.mean():.4f}    "
              f"{gap.mean():.2f}%   {t_mdl:.1f}s   {t_ort:.1f}s")

        all_model.append(dist_model)
        all_ortools.append(dist_ortools)

        rows.append({
            "timestamp":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "checkpoint":     args.checkpoint,
            "cluster":        cid,
            "n_instances":    n_instances,
            "model_mean":     f"{dist_model.mean():.4f}",
            "model_std":      f"{dist_model.std():.4f}",
            "ortools_mean":   f"{dist_ortools.mean():.4f}",
            "ortools_std":    f"{dist_ortools.std():.4f}",
            "gap_mean":       f"{gap.mean():.2f}",
            "gap_std":        f"{gap.std():.2f}",
            "t_model_s":      f"{t_mdl:.2f}",
            "t_ortools_s":    f"{t_ort:.2f}",
            "time_limit_s":   args.time_limit,
        })

    # Aggregate
    if all_model:
        all_m  = torch.cat(all_model)
        all_o  = torch.cat(all_ortools)
        all_g  = gap_percent(all_m, all_o)
        print("-" * len(header))
        print(f"  {'OVERALL':<8}  {len(all_m):>4}  "
              f"{all_m.mean():.4f}    {all_o.mean():.4f}    "
              f"{all_g.mean():.2f}%")

    # Save CSV
    if rows:
        write_header = not log_path.exists()
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults saved -> {log_path}")


if __name__ == "__main__":
    main()
