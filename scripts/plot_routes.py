"""
Generate side-by-side Nazari vs OR-Tools route comparison plots.
Picks the 3 instances whose gap is closest to the median gap.

Output:
    docs/images/vrp10_routes.png
    docs/images/vrp20_routes.png

Run:
    uv run scripts/plot_routes.py
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.agent import NazariAgent
from src.environment import VRPEnvironment
from src.trainer import rollout
from src.utils import gap_percent
from src.plot import plot_route

DEVICE   = "cpu"
OUT_DIR  = Path("docs/images")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    dict(
        eval_set    = "eval_sets/vrp10_cap20_n32.pt",
        checkpoint  = "checkpoints/vrp10_cap20_kool/epoch_10000.pt",
        embed_dim   = 128,
        label       = "VRP10",
        out         = OUT_DIR / "vrp10_routes.png",
    ),
    dict(
        eval_set    = "eval_sets/vrp20_cap30_n32.pt",
        checkpoint  = "checkpoints/vrp20_cap30_kool/epoch_20000.pt",
        embed_dim   = 128,
        label       = "VRP20",
        out         = OUT_DIR / "vrp20_routes.png",
    ),
]

N_EXAMPLES = 3   # instances to show per problem size


for cfg in CONFIGS:
    # ── Load eval set ────────────────────────────────────────────────────────
    data = torch.load(cfg["eval_set"], map_location=DEVICE, weights_only=False)
    coords          = data["coords"].to(DEVICE)          # (B, N+1, 2)
    demands         = data["demands"].to(DEVICE)         # (B, N+1)
    dist_ortools    = data["dist_ortools"].to(DEVICE)    # (B,)
    ortools_tours   = data["ortools_tours"]              # list[list[int]]
    vehicle_cap     = float(data["vehicle_capacity"])

    # ── Run Nazari ───────────────────────────────────────────────────────────
    agent = NazariAgent(cfg["checkpoint"], embed_dim=cfg["embed_dim"], device=DEVICE)
    env   = VRPEnvironment(coords, demands, vehicle_capacity=vehicle_cap)

    torch.manual_seed(42)
    with torch.no_grad():
        _, rewards = rollout(agent, env, greedy=True)

    dist_nazari = -rewards                               # (B,)

    # ── Pick N_EXAMPLES closest to median gap ────────────────────────────────
    gap    = gap_percent(dist_nazari, dist_ortools)      # (B,)
    median = gap.median().item()
    diffs  = (gap - median).abs()
    idxs   = diffs.argsort()[:N_EXAMPLES].tolist()

    print(f"\n{cfg['label']} - selected instances {idxs}")
    for i in idxs:
        print(f"  [{i}] Nazari={dist_nazari[i]:.4f}  OR-Tools={dist_ortools[i]:.4f}"
              f"  gap={gap[i]:.1f}%")

    # ── Build Nazari tours from env.tour ─────────────────────────────────────
    def get_nazari_tour(env, batch_idx):
        return [0] + [t[batch_idx].item() for t in env.tour]

    # ── Plot: N_EXAMPLES rows × 2 cols ───────────────────────────────────────
    fig, axes = plt.subplots(N_EXAMPLES, 2,
                             figsize=(12, 5 * N_EXAMPLES),
                             squeeze=False)
    fig.patch.set_facecolor("#f8f9fa")

    for row, idx in enumerate(idxs):
        coords_np = coords[idx].cpu().numpy()   # (N+1, 2)

        nazari_tour  = get_nazari_tour(env, idx)
        ortools_tour = ortools_tours[idx]

        # Left: Nazari
        ax_l = axes[row][0]
        plot_route(
            coords_np, nazari_tour, ax=ax_l,
            title=f"Nazari (greedy)   dist = {dist_nazari[idx]:.3f}",
        )

        # Right: OR-Tools
        ax_r = axes[row][1]
        plot_route(
            coords_np, ortools_tour, ax=ax_r,
            title=f"OR-Tools   dist = {dist_ortools[idx]:.3f}   "
                  f"gap = {gap[idx]:.1f}%",
        )

    fig.suptitle(
        f"{cfg['label']} — Nazari vs OR-Tools  "
        f"(instances closest to median gap {median:.1f}%)",
        fontsize=14, y=1.01,
    )
    plt.tight_layout()
    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {cfg['out']}")
