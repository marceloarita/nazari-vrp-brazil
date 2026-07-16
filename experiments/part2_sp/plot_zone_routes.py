"""
Part II — 3-column route comparison for one SP zone (held-out instances):
    Nazari greedy | Nazari best-of-N | OR-Tools

Uses the zone-trained model and the leakage-free held-out eval set.
OR-Tools tours are computed once and cached back into the eval .pt.

Output:
    artifacts/figures/c{cluster}_zone_routes.png

Run:
    uv run experiments/part2_sp/plot_zone_routes.py --cluster 5 --samples 128
"""

import argparse
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.solver import solve_batch
from src.trainer import rollout
from src.utils import gap_percent
from src.plot import plot_route

OUT_DIR = Path("artifacts/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_EXAMPLES = 3


def instance_tour(env, i):
    return [0] + [t[i].item() for t in env.tour]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", type=int, default=5)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--time_limit", type=int, default=10, help="OR-Tools s/instance")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--checkpoint", type=str, default=None,
                   help="default: zone model vrp20_real_c{cluster}__kool__base; "
                        "pass the uniform checkpoint to plot the zero-shot baseline")
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    c = args.cluster
    device = "cpu"

    eval_path = next(Path("artifacts/instances").glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt"))
    ckpt = args.checkpoint or f"artifacts/checkpoints/vrp20_real_c{c}__kool__base/epoch_20000.pt"

    data    = torch.load(eval_path, map_location=device, weights_only=False)
    coords  = data["coords"].to(device)
    demands = data["demands"].to(device)
    cap     = float(data["vehicle_capacity"])
    B       = coords.size(0)
    raw_dem = (demands * 30).round().cpu().numpy()   # SP VRP20 capacity=30; recover raw demand 1-9 for labels

    # --- OR-Tools tours (cache into the .pt for reuse) ---
    if "ortools_tours" in data:
        ortools_tours = data["ortools_tours"]
        dist_ortools  = data["dist_ortools"].to(device)
        print("OR-Tools: using cached tours")
    else:
        print(f"OR-Tools: solving {B} instances ({args.time_limit}s each)...")
        t0 = time.time()
        ortools_tours, dist_ortools = solve_batch(coords, demands, vehicle_capacity=cap,
                                                  time_limit_s=args.time_limit)
        print(f"  done in {time.time()-t0:.0f}s")
        data["ortools_tours"] = ortools_tours
        data["dist_ortools"]  = dist_ortools.cpu()
        torch.save(data, eval_path)

    agent = NazariAgent(ckpt, embed_dim=args.embed_dim, device=device)

    # --- Greedy ---
    env_g = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    with torch.no_grad():
        _, r_g = rollout(agent, env_g, greedy=True)
    dist_greedy  = -r_g
    greedy_tours = [instance_tour(env_g, i) for i in range(B)]

    # --- Best-of-N ---
    env_s = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    best_reward = torch.full((B,), float("-inf"))
    best_tours  = [None] * B
    with torch.no_grad():
        for _ in range(args.samples):
            _, r = rollout(agent, env_s, greedy=False)
            for i in range(B):
                if r[i] > best_reward[i]:
                    best_reward[i] = r[i]
                    best_tours[i]  = instance_tour(env_s, i)
    dist_best = -best_reward

    gap_g = gap_percent(dist_greedy, dist_ortools)
    gap_b = gap_percent(dist_best,   dist_ortools)

    median = gap_g.median().item()
    idxs   = (gap_g - median).abs().argsort()[:N_EXAMPLES].tolist()

    print(f"\nC{c} - instances {idxs} (median greedy gap {median:.1f}%)")
    for i in idxs:
        print(f"  [{i}] greedy={dist_greedy[i]:.3f} ({gap_g[i]:.1f}%)  "
              f"best-{args.samples}={dist_best[i]:.3f} ({gap_b[i]:.1f}%)  "
              f"or-tools={dist_ortools[i]:.3f}")

    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(16, 5 * N_EXAMPLES), squeeze=False)
    fig.patch.set_facecolor("#f8f9fa")
    for row, idx in enumerate(idxs):
        cn = coords[idx].cpu().numpy()
        # Square zoom box centered on this instance's points (undistorted, no wasted margin)
        xmin, ymin = cn.min(axis=0)
        xmax, ymax = cn.max(axis=0)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half = max(xmax - xmin, ymax - ymin) / 2 * 1.12  # 12% padding
        cols = [
            (greedy_tours[idx],  f"Nazari greedy   dist = {dist_greedy[idx]:.3f}   gap = {gap_g[idx]:.1f}%"),
            (best_tours[idx],    f"Nazari best-{args.samples}   dist = {dist_best[idx]:.3f}   gap = {gap_b[idx]:.1f}%"),
            (ortools_tours[idx], f"OR-Tools   dist = {dist_ortools[idx]:.3f}"),
        ]
        for col, (tour, title) in enumerate(cols):
            ax = axes[row][col]
            plot_route(cn, tour, ax=ax, title=title, demands=raw_dem[idx])
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"SP zone C{c} (held-out) — Nazari greedy vs best-of-{args.samples} vs OR-Tools "
        f"(near median greedy gap {median:.1f}%)",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    out = Path(args.out) if args.out else OUT_DIR / f"c{c}_zone_routes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
