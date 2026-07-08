import os

import torch
import numpy as np
import matplotlib.pyplot as plt


def euclidean_dist(a, b):
    """Euclidean distance between tensors a and b (last dim must be 2)."""
    return torch.norm(a - b, dim=-1)


def tour_length(tour_indices, coords):
    """
    Compute total tour length from a sequence of visited node indices.

    Args:
        tour_indices: list of (B,) tensors, or a (T, B) tensor
        coords:       (B, N+1, 2) — node coordinates

    Returns:
        (B,) tensor — total Euclidean distance
    """
    if isinstance(tour_indices, list):
        tour = torch.stack(tour_indices, dim=0)  # (T, B)
    else:
        tour = tour_indices                       # (T, B)

    T, B = tour.shape
    total = torch.zeros(B, device=coords.device)
    for t in range(1, T):
        prev = coords[torch.arange(B), tour[t - 1]]  # (B, 2)
        curr = coords[torch.arange(B), tour[t]]       # (B, 2)
        total += torch.norm(curr - prev, dim=-1)
    return total


def save_checkpoint(path, actor, critic, actor_opt, critic_opt, epoch):
    """Save model and optimizer states."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "actor_state": actor.state_dict(),
            "critic_state": critic.state_dict(),
            "actor_opt_state": actor_opt.state_dict(),
            "critic_opt_state": critic_opt.state_dict(),
        },
        path,
    )


def load_checkpoint(path, actor, critic, actor_opt=None, critic_opt=None):
    """
    Load model and optimizer states from a checkpoint file.

    Returns:
        epoch: int — epoch at which the checkpoint was saved
    """
    ckpt = torch.load(path, map_location="cpu")
    actor.load_state_dict(ckpt["actor_state"])
    critic.load_state_dict(ckpt["critic_state"])
    if actor_opt is not None:
        actor_opt.load_state_dict(ckpt["actor_opt_state"])
    if critic_opt is not None:
        critic_opt.load_state_dict(ckpt["critic_opt_state"])
    return ckpt["epoch"]


def plot_routes(coords, tour_indices, title=None, ax=None):
    """
    Visualize VRP routes for a single instance.

    Args:
        coords:       (N+1, 2) numpy array or tensor — index 0 = depot
        tour_indices: list of ints — full sequence of visited node indices
        title:        optional plot title
        ax:           optional matplotlib Axes (creates a new figure if None)

    Returns:
        ax: matplotlib Axes
    """
    if hasattr(coords, "cpu"):
        coords = coords.cpu().numpy()
    tour = [i.item() if hasattr(i, "item") else int(i) for i in tour_indices]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(coords[1:, 0], coords[1:, 1], c="steelblue", s=40, zorder=3, label="Customers")
    ax.scatter(coords[0, 0], coords[0, 1], c="crimson", s=150, marker="*", zorder=4, label="Depot")

    colors = plt.cm.tab10.colors
    color_idx = 0
    segment = [0]
    for node in tour[1:]:
        segment.append(node)
        if node == 0:
            xs = coords[segment, 0]
            ys = coords[segment, 1]
            ax.plot(xs, ys, c=colors[color_idx % len(colors)], lw=1.5)
            color_idx += 1
            segment = [0]

    ax.set_aspect("equal")
    ax.legend(loc="best")
    if title:
        ax.set_title(title)
    return ax


def gap_percent(model_dist, solver_dist):
    """
    Gap% = (model_dist - solver_dist) / solver_dist * 100.
    Lower is better. Used to compare against OR-Tools baseline.
    """
    return (model_dist - solver_dist) / solver_dist * 100.0
