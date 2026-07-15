"""
Generate 3-column route comparisons: Nazari greedy | Nazari best-of-N | OR-Tools.
Picks the 3 instances whose greedy gap is closest to the median.

Output:
    docs/images/vrp10_routes.png
    docs/images/vrp20_routes.png

Run:
    uv run scripts/plot_routes.py
"""

import torch
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

N_EXAMPLES = 3       # instances (rows) per problem size
SAMPLES    = 128     # best-of-N
SEED       = 42

CONFIGS = [
    dict(
        eval_set   = "artifacts/instances/vrp10_cap20_n32.pt",
        checkpoint = "artifacts/checkpoints/vrp10_cap20_kool/epoch_10000.pt",
        embed_dim  = 128,
        label      = "VRP10",
        out        = OUT_DIR / "vrp10_routes.png",
    ),
    dict(
        eval_set   = "artifacts/instances/vrp20_cap30_n32.pt",
        checkpoint = "artifacts/checkpoints/vrp20_cap30_kool/epoch_20000.pt",
        embed_dim  = 128,
        label      = "VRP20",
        out        = OUT_DIR / "vrp20_routes.png",
    ),
]


def instance_tour(env, i):
    """Reconstruct the full visit sequence for instance i from env.tour."""
    return [0] + [t[i].item() for t in env.tour]


for cfg in CONFIGS:
    torch.manual_seed(SEED)

    data          = torch.load(cfg["eval_set"], map_location=DEVICE, weights_only=False)
    coords        = data["coords"].to(DEVICE)
    demands       = data["demands"].to(DEVICE)
    dist_ortools  = data["dist_ortools"].to(DEVICE)
    ortools_tours = data["ortools_tours"]
    cap           = float(data["vehicle_capacity"])
    B             = coords.size(0)

    agent = NazariAgent(cfg["checkpoint"], embed_dim=cfg["embed_dim"], device=DEVICE)

    # --- Greedy ---
    env_g = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    with torch.no_grad():
        _, r_greedy = rollout(agent, env_g, greedy=True)
    dist_greedy   = -r_greedy
    greedy_tours  = [instance_tour(env_g, i) for i in range(B)]

    # --- Best-of-N sampling: keep the best tour per instance ---
    env_s = VRPEnvironment(coords, demands, vehicle_capacity=cap)
    best_reward = torch.full((B,), float("-inf"))
    best_tours  = [None] * B
    with torch.no_grad():
        for _ in range(SAMPLES):
            _, r = rollout(agent, env_s, greedy=False)
            for i in range(B):
                if r[i] > best_reward[i]:
                    best_reward[i] = r[i]
                    best_tours[i]  = instance_tour(env_s, i)
    dist_best = -best_reward

    gap_greedy = gap_percent(dist_greedy, dist_ortools)
    gap_best   = gap_percent(dist_best,   dist_ortools)

    # --- Pick N_EXAMPLES closest to the median greedy gap ---
    median = gap_greedy.median().item()
    idxs   = (gap_greedy - median).abs().argsort()[:N_EXAMPLES].tolist()

    print(f"\n{cfg['label']} - instances {idxs} (median greedy gap {median:.1f}%)")
    for i in idxs:
        print(f"  [{i}] greedy={dist_greedy[i]:.3f} ({gap_greedy[i]:.1f}%)  "
              f"best-{SAMPLES}={dist_best[i]:.3f} ({gap_best[i]:.1f}%)  "
              f"or-tools={dist_ortools[i]:.3f}")

    # --- Plot: N_EXAMPLES rows x 3 cols ---
    fig, axes = plt.subplots(N_EXAMPLES, 3, figsize=(16, 5 * N_EXAMPLES), squeeze=False)
    fig.patch.set_facecolor("#f8f9fa")

    for row, idx in enumerate(idxs):
        coords_np = coords[idx].cpu().numpy()
        plot_route(coords_np, greedy_tours[idx], ax=axes[row][0],
                   title=f"Nazari greedy   dist = {dist_greedy[idx]:.3f}   gap = {gap_greedy[idx]:.1f}%")
        plot_route(coords_np, best_tours[idx], ax=axes[row][1],
                   title=f"Nazari best-{SAMPLES}   dist = {dist_best[idx]:.3f}   gap = {gap_best[idx]:.1f}%")
        plot_route(coords_np, ortools_tours[idx], ax=axes[row][2],
                   title=f"OR-Tools   dist = {dist_ortools[idx]:.3f}")

    fig.suptitle(
        f"{cfg['label']} — Nazari greedy vs best-of-{SAMPLES} vs OR-Tools "
        f"(instances near median greedy gap {median:.1f}%)",
        fontsize=14, y=1.005,
    )
    plt.tight_layout()
    fig.savefig(cfg["out"], dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {cfg['out']}")
