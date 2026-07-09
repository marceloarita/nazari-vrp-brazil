"""
OR-Tools CVRP solver — golden truth baseline.

Solves each instance to optimality (or near-optimality with time limit)
using Google OR-Tools' vehicle routing library.
"""

import numpy as np
import torch
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def _solve_instance(coords_np, demands_np, vehicle_capacity, time_limit_s=10):
    """
    Solve a single CVRP instance with OR-Tools.

    Args:
        coords_np:        (N+1, 2) float — node coordinates; index 0 = depot
        demands_np:       (N+1,)   float — normalized demands; depot = 0.0
        vehicle_capacity: float — normalized vehicle capacity (typically 1.0)
        time_limit_s:     int   — solver time limit in seconds

    Returns:
        tour:     list of ints — full sequence of visited nodes (depot-to-depot)
        distance: float — total Euclidean distance
    """
    n_nodes = len(coords_np)

    # OR-Tools works with integers — scale distances to avoid precision loss
    scale = 10_000

    def dist(i, j):
        dx = coords_np[i, 0] - coords_np[j, 0]
        dy = coords_np[i, 1] - coords_np[j, 1]
        return int(np.sqrt(dx**2 + dy**2) * scale)

    # Demand must be integer; scale and round
    demand_scale = 1_000
    demands_int = [int(d * demand_scale) for d in demands_np]
    capacity_int = int(vehicle_capacity * demand_scale)

    manager = pywrapcp.RoutingIndexManager(n_nodes, n_nodes, 0)  # up to n_nodes vehicles
    routing = pywrapcp.RoutingModel(manager)

    # Distance callback
    def distance_callback(from_index, to_index):
        return dist(manager.IndexToNode(from_index), manager.IndexToNode(to_index))

    transit_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Capacity constraint
    def demand_callback(from_index):
        return demands_int[manager.IndexToNode(from_index)]

    demand_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, [capacity_int] * n_nodes, True, "Capacity"
    )

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = time_limit_s

    solution = routing.SolveWithParameters(params)

    if solution is None:
        return None, float("inf")

    # Extract tour as flat sequence (depot visits included)
    tour = []
    total_dist = 0.0
    for v in range(n_nodes):
        idx = routing.Start(v)
        if routing.IsEnd(solution.Value(routing.NextVar(idx))):
            continue  # unused vehicle
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            tour.append(node)
            next_idx = solution.Value(routing.NextVar(idx))
            next_node = manager.IndexToNode(next_idx)
            total_dist += np.sqrt(
                (coords_np[node, 0] - coords_np[next_node, 0]) ** 2 +
                (coords_np[node, 1] - coords_np[next_node, 1]) ** 2
            )
            idx = next_idx
        tour.append(0)  # return to depot

    return tour, total_dist


def solve_batch(coords, demands, vehicle_capacity=1.0, time_limit_s=10):
    """
    Solve a batch of CVRP instances with OR-Tools.

    Args:
        coords:           (B, N+1, 2) tensor
        demands:          (B, N+1)    tensor — normalized demands
        vehicle_capacity: float
        time_limit_s:     int — per-instance time limit

    Returns:
        tours:     list of B tour lists
        distances: (B,) tensor — total distance per instance
    """
    B = coords.size(0)
    coords_np  = coords.cpu().numpy()
    demands_np = demands.cpu().numpy()

    tours = []
    distances = []

    for b in range(B):
        tour, dist = _solve_instance(
            coords_np[b], demands_np[b], vehicle_capacity, time_limit_s
        )
        tours.append(tour)
        distances.append(dist)
        if (b + 1) % 10 == 0:
            print(f"  OR-Tools: {b+1}/{B} solved")

    return tours, torch.tensor(distances, dtype=torch.float32, device=coords.device)
