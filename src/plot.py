import numpy as np
import matplotlib.pyplot as plt

ROUTE_COLORS = ["#5c85d6", "#4aab8f", "#c96b4a", "#7b62b8", "#c9943a", "#b85c7a"]


def plot_route(coords, tour, ax=None, title=None):
    """
    Plot a single VRP route.

    Args:
        coords: (N+1, 2) numpy array — index 0 = depot
        tour:   list of ints — full visit sequence, e.g. [0, 3, 5, 0, 2, 0]
        ax:     matplotlib Axes (creates a new figure if None)
        title:  optional string

    Returns:
        ax: matplotlib Axes
    """
    if hasattr(coords, "cpu"):
        coords = coords.cpu().numpy()

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))

    ax.set_facecolor("#f8f9fa")

    # Route segments
    if tour:
        segment = [0]
        color_idx = 0
        for node in tour:
            if node == 0:
                if len(segment) > 1:
                    segment.append(0)  # close route back to depot
                    ax.plot(coords[segment, 0], coords[segment, 1],
                            color=ROUTE_COLORS[color_idx % len(ROUTE_COLORS)],
                            lw=2.0, alpha=0.8, zorder=1)
                    color_idx += 1
                segment = [0]
            else:
                segment.append(node)
        # draw any open segment that didn't close at depot
        if len(segment) > 1:
            segment.append(0)
            ax.plot(coords[segment, 0], coords[segment, 1],
                    color=ROUTE_COLORS[color_idx % len(ROUTE_COLORS)],
                    lw=2.0, alpha=0.8, zorder=1)

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

    # Axes style
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(labelsize=8, color="#b2bec3")
    ax.set_xlabel("X", fontsize=10, color="#636e72")
    ax.set_ylabel("Y", fontsize=10, color="#636e72")
    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    if title:
        ax.set_title(title, fontsize=12, color="#2d3436", pad=10)

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
