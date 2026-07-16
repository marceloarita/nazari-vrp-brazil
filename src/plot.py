import numpy as np
import matplotlib.pyplot as plt

ROUTE_COLORS = ["#e74c3c", "#27ae60", "#2980b9", "#00bcd4", "#9b59b6", "#e67e22"]


def plot_route(coords, tour, ax=None, title=None, demands=None):
    """
    Plot a single VRP route.

    Args:
        coords:  (N+1, 2) numpy array — index 0 = depot
        tour:    list of ints — full visit sequence, e.g. [0, 3, 5, 0, 2, 0]
        ax:      matplotlib Axes (creates a new figure if None)
        title:   optional string
        demands: optional (N+1,) array — if given, each customer is labelled with its
                 demand (raw integer) instead of its node index; depot is unlabelled.

    Returns:
        ax: matplotlib Axes
    """
    if hasattr(coords, "cpu"):
        coords = coords.cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.set_facecolor("white")

    # Route segments
    if tour:
        segment = [0]
        color_idx = 0
        for node in tour:
            if node == 0:
                if len(segment) > 1:
                    segment.append(0)
                    ax.plot(coords[segment, 0], coords[segment, 1],
                            color=ROUTE_COLORS[color_idx % len(ROUTE_COLORS)],
                            lw=2.5, zorder=1)
                    color_idx += 1
                segment = [0]
            else:
                segment.append(node)
        # draw any open segment that didn't close at depot
        if len(segment) > 1:
            segment.append(0)
            ax.plot(coords[segment, 0], coords[segment, 1],
                    color=ROUTE_COLORS[color_idx % len(ROUTE_COLORS)],
                    lw=2.5, zorder=1)

    # Customer nodes — pastel Claude orange
    for i in range(1, len(coords)):
        ax.scatter(coords[i, 0], coords[i, 1],
                   s=300, color="#F5C49A", edgecolors="#F5C49A",
                   linewidth=1.5, zorder=3)
        label = str(int(round(float(demands[i])))) if demands is not None else str(i)
        ax.text(coords[i, 0], coords[i, 1], label,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1a1a1a", zorder=4)

    # Depot — Claude orange
    ax.scatter(coords[0, 0], coords[0, 1],
               s=280, marker="s", color="#D4651C",
               edgecolors="#D4651C", linewidth=1.5, zorder=5)

    # Axes style
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=11, color="#555555")
    ax.set_xlabel("X", fontsize=13, color="#333333")
    ax.set_ylabel("Y", fontsize=13, color="#333333")
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(1.0)

    if title:
        ax.set_title(title, fontsize=12, color="#1a1a1a", pad=10)

    return ax


def plot_comparison(entries, figsize=None):
    """
    Plot multiple VRP routes side by side.

    Args:
        entries: list of dicts, each with keys:
                   coords (N+1,2), tour (list of ints), title (str)
        figsize: optional (width, height) — defaults to (7*n, 7)
    """
    n = len(entries)
    if figsize is None:
        figsize = (7 * n, 7)

    fig, axes = plt.subplots(1, n, figsize=figsize)
    fig.patch.set_facecolor("#f8f9fa")

    if n == 1:
        axes = [axes]

    for ax, entry in zip(axes, entries):
        coords = entry["coords"]
        if hasattr(coords, "cpu"):
            coords = coords.cpu().numpy()
        plot_route(coords, entry["tour"], ax=ax, title=entry.get("title"))

    plt.tight_layout()
    return fig
