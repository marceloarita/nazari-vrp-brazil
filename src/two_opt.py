"""
2-opt local search for CVRP tours (Part II, Hypothesis A4).

Post-processing applied to a trained model's output — no retraining. 2-opt is
run *within each trip* (each depot→…→depot segment): reordering customers inside
a trip never changes the trip's total demand, so feasibility is preserved, and
uncrossing two edges always shortens the path (triangle inequality). This removes
the self-crossings ("X") the greedy policy leaves behind.
"""

import numpy as np


def _d(coords, a, b):
    return float(np.linalg.norm(coords[a] - coords[b]))


def tour_length(coords, tour):
    """Total Euclidean length of a full visit sequence (node indices; 0 = depot)."""
    coords = np.asarray(coords)
    return sum(_d(coords, tour[k], tour[k + 1]) for k in range(len(tour) - 1))


def _two_opt_trip(coords, route, eps=1e-9):
    """2-opt a single trip route = [0, c1, ..., ck, 0] until no improving swap."""
    route = list(route)
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                a, b = route[i - 1], route[i]
                c, e = route[j], route[j + 1]
                # replace edges (a,b) and (c,e) with (a,c) and (b,e), reversing b..c
                delta = (_d(coords, a, c) + _d(coords, b, e)) - (_d(coords, a, b) + _d(coords, c, e))
                if delta < -eps:
                    route[i:j + 1] = route[i:j + 1][::-1]
                    improved = True
    return route


def two_opt(coords, tour):
    """
    Apply intra-trip 2-opt to a full CVRP tour.

    Args:
        coords: (N+1, 2) array — node coordinates; index 0 = depot
        tour:   list of node indices with depot (0) returns, e.g. [0, 3, 5, 0, 2, 7, 0]

    Returns:
        improved full tour (list of node indices)
    """
    coords = np.asarray(coords)
    trips, cur = [], []
    for node in tour:
        if node == 0:
            if cur:
                trips.append(cur); cur = []
        else:
            cur.append(node)
    if cur:
        trips.append(cur)

    new_tour = [0]
    for trip in trips:
        r = _two_opt_trip(coords, [0] + trip + [0])
        new_tour += r[1:]          # append [c..., 0], skipping the leading depot
    return new_tour
