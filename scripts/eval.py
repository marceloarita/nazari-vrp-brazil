"""
Evaluation script — compares NazariAgent vs RandomAgent on the same instances.

Usage:
    uv run scripts/eval.py --checkpoint checkpoints/epoch_01000.pt --n_customers 20
    uv run scripts/eval.py --checkpoint checkpoints/epoch_01000.pt --n_customers 20 --plot
"""

import argparse
import torch
import matplotlib.pyplot as plt

from src.environment import generate_batch, VRPEnvironment
from src.agent import RandomAgent, NazariAgent
from src.trainer import rollout
from src.utils import gap_percent


def plot_comparison(env_random, env_nazari, batch_idx=0):
    """Plot side-by-side routes for Random vs Nazari on the same instance."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor("#f8f9fa")

    dist_random = env_random.total_dist[batch_idx].item()
    dist_nazari = env_nazari.total_dist[batch_idx].item()

    for ax, env, label, dist in [
        (axes[0], env_random, "Random Agent",        dist_random),
        (axes[1], env_nazari, "Nazari Agent (greedy)", dist_nazari),
    ]:
        coords = env.static[batch_idx].cpu().numpy()
        route_colors = ["#5c85d6", "#4aab8f", "#c96b4a", "#7b62b8", "#c9943a", "#b85c7a"]

        ax.set_facecolor("#f8f9fa")

        # Route segments
        if env.tour:
            tour = [0] + [t[batch_idx].item() for t in env.tour]
            segment = [0]
            color_idx = 0
            for node in tour[1:]:
                segment.append(node)
                if node == 0:
                    ax.plot(coords[segment, 0], coords[segment, 1],
                            color=route_colors[color_idx % len(route_colors)],
                            lw=2.0, alpha=0.8, zorder=1)
                    color_idx += 1
                    segment = [0]

        # Customer nodes
        for i in range(1, len(coords)):
            ax.scatter(coords[i, 0], coords[i, 1],
                       s=260, color="#ffeaa7", edgecolors="#fdcb6e",
                       linewidth=1.2, zorder=3)
            ax.text(coords[i, 0], coords[i, 1], str(i),
                    ha="center", va="center",
                    fontsize=9, fontweight="bold", color="#2d3436", zorder=4)

        # Depot
        ax.scatter(coords[0, 0], coords[0, 1],
                   s=200, marker="s", color="#ff7675",
                   edgecolors="#d63031", linewidth=1.5, zorder=5)
        ax.text(coords[0, 0], coords[0, 1], "D",
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="white", zorder=6)

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(labelsize=8, color="#b2bec3")
        for spine in ax.spines.values():
            spine.set_color("#dfe6e9")
            spine.set_linewidth(1.0)

        ax.set_title(f"{label}\ndist: {dist:.4f}", fontsize=12, color="#2d3436", pad=10)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to NazariAgent checkpoint")
    parser.add_argument("--n_customers", type=int, default=20)
    parser.add_argument("--n_instances", type=int, default=256)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--plot", action="store_true", help="Plot routes for first instance")
    parser.add_argument("--plot_idx", type=int, default=0, help="Batch index to plot")
    args = parser.parse_args()

    device = args.device
    torch.manual_seed(42)

    print(f"VRP{args.n_customers} | {args.n_instances} instances | device: {device}\n")

    # --- Generate shared instances ---
    coords, demands = generate_batch(args.n_instances, args.n_customers, device=device)

    # --- Random baseline ---
    random_agent = RandomAgent()
    env_random = VRPEnvironment(coords, demands)
    with torch.no_grad():
        _, rewards_random = rollout(random_agent, env_random, greedy=False)

    # --- Nazari agent ---
    nazari_agent = NazariAgent(args.checkpoint, embed_dim=args.embed_dim, device=device)
    env_nazari = VRPEnvironment(coords, demands)
    with torch.no_grad():
        _, rewards_nazari = rollout(nazari_agent, env_nazari, greedy=True)

    # --- Results ---
    dist_random = -rewards_random
    dist_nazari = -rewards_nazari
    gap = gap_percent(dist_nazari, dist_random)

    print(f"{'':20s}  {'mean':>8s}  {'std':>8s}  {'min':>8s}  {'max':>8s}")
    print(f"{'Random':20s}  {dist_random.mean():8.4f}  {dist_random.std():8.4f}  {dist_random.min():8.4f}  {dist_random.max():8.4f}")
    print(f"{'Nazari (greedy)':20s}  {dist_nazari.mean():8.4f}  {dist_nazari.std():8.4f}  {dist_nazari.min():8.4f}  {dist_nazari.max():8.4f}")
    print(f"\nGap% (Nazari vs Random): {gap.mean():.2f}% (negative = Nazari is better)")

    if args.plot:
        plot_comparison(env_random, env_nazari, batch_idx=args.plot_idx)


if __name__ == "__main__":
    main()
