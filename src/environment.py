import numpy as np
import torch
import matplotlib.pyplot as plt
from .plot import plot_route

DEPOT_IDX = 0


def generate_instance(n_customers, vehicle_capacity=1.0, distribution="uniform", kde=None):
    """
    Returns:
        coords:  (n_customers+1, 2) float32, values in [0,1]; index 0 = depot
        demands: (n_customers+1,) float32, normalized by vehicle_capacity; depot = 0.0
    """
    if distribution == "uniform":
        coords = np.random.uniform(0, 1, (n_customers + 1, 2)).astype(np.float32)
    elif distribution == "kde":
        if kde is None:
            raise ValueError("KDE object required for 'kde' distribution")
        customer_xy = kde.sample(n_customers).astype(np.float32)
        depot_xy = np.random.uniform(0, 1, (1, 2)).astype(np.float32)
        coords = np.vstack([depot_xy, customer_xy])
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    raw_demands = np.random.randint(1, 10, n_customers).astype(np.float32)
    demands = np.concatenate([np.zeros(1, dtype=np.float32), raw_demands / vehicle_capacity])
    return coords, demands


def generate_batch(batch_size, n_customers, vehicle_capacity=1.0, distribution="uniform", kde=None, device="cpu"):
    """
    Returns tensors ready for the model:
        coords:  (B, n_customers+1, 2)
        demands: (B, n_customers+1)

    For uniform distribution, tensors are generated directly on `device` (avoids CPU→GPU transfer).
    KDE distribution falls back to numpy (CPU only).
    """
    if distribution == "kde":
        instances = [
            generate_instance(n_customers, vehicle_capacity, distribution, kde)
            for _ in range(batch_size)
        ]
        coords = torch.tensor(np.stack([c for c, _ in instances]), device=device)
        demands = torch.tensor(np.stack([d for _, d in instances]), device=device)
        return coords, demands

    # Uniform: generate directly on target device
    N = n_customers + 1
    coords = torch.rand(batch_size, N, 2, device=device)
    raw_demands = torch.randint(1, 10, (batch_size, n_customers), device=device).float()
    depot_demand = torch.zeros(batch_size, 1, device=device)
    demands = torch.cat([depot_demand, raw_demands / vehicle_capacity], dim=1)
    return coords, demands

class VRPEnvironment:
    """
    Batched CVRP environment. Maintains episode state and enforces the masking scheme.

    State representation:
      static:  (B, N+1, 2) — node coordinates (fixed throughout episode)
      dynamic: (B, N+1, 2) — [remaining_demand, remaining_capacity] (updated each step)
      mask:    (B, N+1)    — True = node is forbidden at the current step

    Masking rules (Nazari et al. 2018):
      Rule 1: customer with demand == 0 is masked (already served)
      Rule 2: if all customers are unreachable, only depot remains (implicit — depot never masked)
      Rule 3: customer with demand > remaining capacity is masked
    """

    def __init__(self, coords, demands, vehicle_capacity=1.0):
        self.B, self.n_nodes = demands.shape
        self.device = coords.device
        self.static = coords
        self._init_demands = demands.clone()
        self.vehicle_capacity = vehicle_capacity
        self._reset_buffers()

    def _reset_buffers(self):
        self.demands = self._init_demands.clone()
        self.remaining_cap = torch.ones(self.B, device=self.device)
        self.current_node = torch.zeros(self.B, dtype=torch.long, device=self.device)
        self.total_dist = torch.zeros(self.B, device=self.device)
        # Tour history: list of (B,) tensors, one per step
        self.tour: list = []

    def reset(self):
        self._reset_buffers()
        return self.static, self._dynamic(), self._mask()

    def step(self, actions):
        """
        Args:
            actions: (B,) long — index of the next node to visit

        Returns:
            (static, dynamic, mask), reward (B,), done (B,)

        Reward is sparse: non-zero only at the terminal step (all served + at depot).
        """
        B = self.B
        batch = torch.arange(B, device=self.device)

        prev_coords = self.static[batch, self.current_node]
        next_coords = self.static[batch, actions]
        self.total_dist += torch.norm(next_coords - prev_coords, dim=-1)

        is_customer = actions != DEPOT_IDX

        # Consume demand before zeroing it out
        action_demand = self.demands[batch, actions]

        self.remaining_cap = torch.where(
            is_customer,
            self.remaining_cap - action_demand,
            torch.ones_like(self.remaining_cap),  # reload at depot
        )
        self.remaining_cap = self.remaining_cap.clamp(min=0.0)

        # Zero out demand for visited customer; leave depot demand unchanged (already 0)
        self.demands[batch, actions] = torch.where(
            is_customer, torch.zeros_like(action_demand), action_demand
        )

        self.current_node = actions
        self.tour.append(actions.clone())

        all_served = (self.demands[:, 1:] == 0).all(dim=-1)
        done = all_served & ~is_customer  # all served AND returned to depot

        reward = torch.where(done, -self.total_dist, torch.zeros_like(self.total_dist))
        return (self.static, self._dynamic(), self._mask()), reward, done

    def _dynamic(self):
        cap = self.remaining_cap.unsqueeze(-1).expand_as(self.demands)
        return torch.stack([self.demands, cap], dim=-1)  # (B, N+1, 2)

    def _mask(self):
        mask = torch.zeros(self.B, self.n_nodes, dtype=torch.bool, device=self.device)

        # Rule 1: already-served customers
        mask[:, 1:] = self.demands[:, 1:] == 0

        # Rule 3: demand exceeds remaining vehicle capacity
        mask[:, 1:] |= self.demands[:, 1:] > self.remaining_cap.unsqueeze(-1)

        # Depot is masked when vehicle is already at depot (forces visiting a customer first).
        # Exception: if all customers are masked (none reachable), depot must stay open to avoid deadlock.
        at_depot = (self.current_node == DEPOT_IDX)
        any_customer_open = (~mask[:, 1:]).any(dim=-1)  # True if ≥1 customer is reachable
        mask[at_depot & any_customer_open, DEPOT_IDX] = True

        return mask

    def render(self, batch_idx: int = 0, title: str | None = None) -> None:
        """Plot nodes and route for one instance in the batch."""
        coords = self.static[batch_idx].cpu().numpy()
        tour = [0] + [t[batch_idx].item() for t in self.tour] if self.tour else []
        fig, ax = plt.subplots(figsize=(7, 7))
        fig.patch.set_facecolor("#f8f9fa")
        plot_route(coords, tour, ax=ax, title=title)
        plt.tight_layout()
        plt.show()

    @property
    def final_reward(self):
        """Returns -total_dist for all episodes (call after done.all() is True)."""
        return -self.total_dist
