"""
Part II — 3-column progression comparison for one SP zone (held-out instances):

    Zero-shot | Nazari Best(N=128) | POMO Best(N=128) + 2-opt

Same held-out C5 instances for every column, each labelled with its distance and
gap to OR-Tools. Same visual style as the Part I route figures
(docs/images/vrp10_routes.png, vrp20_routes.png).

    Col 1  Zero-shot                 uniform model, best-of-N
    Col 2  Nazari Best(N=128)        per-zone H1 (Kool baseline), best-of-N
    Col 3  POMO Best(N=128) + 2-opt  per-zone POMO, best-of-N then 2-opt

Output:
    artifacts/figures/c{cluster}_progression_routes.png

Run:
    uv run experiments/part2_sp/plot_progression_routes.py --cluster 5 --samples 128
"""

import argparse
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.solver import solve_batch
from src.trainer import rollout
from src.utils import gap_percent
from src.plot import plot_route
from src.two_opt import two_opt, tour_length

OUT_DIR = Path("artifacts/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
N_EXAMPLES = 3


def instance_tour(env, i):
    return [0] + [t[i].item() for t in env.tour]


def best_of_n(agent, coords, demands, cap, samples, B):
    """Best-of-N sampling; returns (dist tensor, list of best tours)."""
    env = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    best_reward = torch.full((B,), float("-inf"))
    best_tours = [None] * B
    with torch.no_grad():
        for _ in range(samples):
            _, r = rollout(agent, env, greedy=False)
            for i in range(B):
                if r[i] > best_reward[i]:
                    best_reward[i] = r[i]
                    best_tours[i] = instance_tour(env, i)
    return -best_reward, best_tours


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cluster", type=int, default=5)
    p.add_argument("--samples", type=int, default=128)
    p.add_argument("--time_limit", type=int, default=10, help="OR-Tools s/instance")
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    c = args.cluster
    device = "cpu"
    S = args.samples

    eval_path = next(Path("artifacts/instances").glob(f"olist_sp_c{c}_vrp20_heldout_n*_seed*.pt"))
    zero_ckpt = "artifacts/checkpoints/vrp20_cap30_kool/epoch_20000.pt"
    h1_ckpt   = f"artifacts/checkpoints/vrp20_real_c{c}__kool__base/epoch_20000.pt"
    pomo_ckpt = f"artifacts/checkpoints/vrp20_real_c{c}_pomo/epoch_20000.pt"

    data    = torch.load(eval_path, map_location=device, weights_only=False)
    coords  = data["coords"].to(device)
    demands = data["demands"].to(device)
    cap     = float(data["vehicle_capacity"])
    B       = coords.size(0)
    raw_dem = (demands * 30).round().cpu().numpy()   # recover raw demand 1-9 for labels
    cn_all  = coords.cpu().numpy()

    # --- OR-Tools tours (cache into the .pt for reuse) ---
    if "dist_ortools" in data:
        dist_ortools = data["dist_ortools"].to(device)
        print("OR-Tools: using cached distances")
    else:
        print(f"OR-Tools: solving {B} instances ({args.time_limit}s each)...")
        t0 = time.time()
        _, dist_ortools = solve_batch(coords, demands, vehicle_capacity=cap,
                                      time_limit_s=args.time_limit)
        print(f"  done in {time.time()-t0:.0f}s")
        data["dist_ortools"] = dist_ortools.cpu()
        torch.save(data, eval_path)

    # --- Zero-shot (uniform) best-of-N ---
    print("Zero-shot best-of-N ...")
    agent_z = NazariAgent(zero_ckpt, embed_dim=args.embed_dim, device=device)
    dist_zero, tours_zero = best_of_n(agent_z, coords, demands, cap, S, B)

    # --- Nazari per-zone (H1) best-of-N ---
    print("Nazari (H1) best-of-N ...")
    agent_h1 = NazariAgent(h1_ckpt, embed_dim=args.embed_dim, device=device)
    dist_h1, tours_h1 = best_of_n(agent_h1, coords, demands, cap, S, B)

    # --- POMO best-of-N, then 2-opt ---
    print("POMO best-of-N + 2-opt ...")
    agent_p = NazariAgent(pomo_ckpt, embed_dim=args.embed_dim, device=device)
    _, tours_pomo = best_of_n(agent_p, coords, demands, cap, S, B)
    tours_pomo2o = [two_opt(cn_all[i], tours_pomo[i]) for i in range(B)]
    dist_pomo2o = torch.tensor([tour_length(cn_all[i], tours_pomo2o[i]) for i in range(B)])

    gap_zero = gap_percent(dist_zero,   dist_ortools)
    gap_h1   = gap_percent(dist_h1,     dist_ortools)
    gap_pomo = gap_percent(dist_pomo2o, dist_ortools)

    # Select example instances near the median H1 gap (middle model = representative)
    median = gap_h1.median().item()
    idxs   = (gap_h1 - median).abs().argsort()[:N_EXAMPLES].tolist()

    print(f"\nC{c} - instances {idxs} (median Nazari gap {median:.1f}%)")
    for i in idxs:
        print(f"  [{i}] zero={dist_zero[i]:.3f} ({gap_zero[i]:.1f}%)  "
              f"nazari={dist_h1[i]:.3f} ({gap_h1[i]:.1f}%)  "
              f"pomo+2opt={dist_pomo2o[i]:.3f} ({gap_pomo[i]:.1f}%)")

    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(16, 5 * N_EXAMPLES), squeeze=False)
    fig.patch.set_facecolor("#f8f9fa")
    for row, idx in enumerate(idxs):
        cn = cn_all[idx]
        xmin, ymin = cn.min(axis=0)
        xmax, ymax = cn.max(axis=0)
        cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        half = max(xmax - xmin, ymax - ymin) / 2 * 1.12  # 12% padding
        cols = [
            (tours_zero[idx],   f"Zero-shot   dist = {dist_zero[idx]:.3f}   gap = {gap_zero[idx]:.1f}%"),
            (tours_h1[idx],     f"Nazari Best(N={S})   dist = {dist_h1[idx]:.3f}   gap = {gap_h1[idx]:.1f}%"),
            (tours_pomo2o[idx], f"POMO Best(N={S}) + 2-opt   dist = {dist_pomo2o[idx]:.3f}   gap = {gap_pomo[idx]:.1f}%"),
        ]
        for col, (tour, title) in enumerate(cols):
            ax = axes[row][col]
            plot_route(cn, tour, ax=ax, title=title, demands=raw_dem[idx])
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"SP zone C{c} (held-out) — Zero-shot vs Nazari per-zone vs POMO + 2-opt "
        f"(near median Nazari gap {median:.1f}%)",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    out = Path(args.out) if args.out else OUT_DIR / f"c{c}_progression_routes.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
