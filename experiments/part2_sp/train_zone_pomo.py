"""
Part II / Learning-signal lever — per-zone training with the POMO baseline.

Preserves Hypothesis 1 (per-zone training, direct real-point sampling, fixed
centroid depot) and swaps the Kool baseline for POMO (Kwon et al. 2020):

  For each instance, roll out N trajectories, each forced to start from a
  different customer, and use the per-instance MEAN reward as the baseline.
  The N starts baseline each other — no frozen copy, no critic. Exploits the
  routing symmetry (a tour can start from any node) and is known to generalize
  better out-of-distribution.

Same model (Nazari) and same data as H1 — only the training loop changes.
Effective batch = BATCH_SIZE × N (one trajectory per start node).

Requires:
    artifacts/instances/olist_sp_c{ID}_train_pool.pt  (from eda/05_split_train_test.py)

Run (Kaggle T4 or local):
    uv run experiments/part2_sp/train_zone_pomo.py
"""

import os
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.trainer import Trainer

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
CLUSTER_ID       = 5          # C5 / Centro-Sul
N_CUSTOMERS      = 20
VEHICLE_CAPACITY = 30
BATCH_SIZE       = 64         # effective batch = 64 × 20 starts = 1280 trajectories/step
N_EPOCHS         = 10_000
SAVE_EVERY       = 1_000
LR               = 1e-4
MAX_GRAD_NORM    = 2.0
BASELINE         = "pomo"

RUN_NAME       = f"vrp{N_CUSTOMERS}_real_c{CLUSTER_ID}_pomo"
CHECKPOINT_DIR = f"artifacts/checkpoints/{RUN_NAME}"
# Same W&B project as H1/H2 (nazari-vrp20-real-c5) for side-by-side comparison.
WANDB_PROJECT  = f"nazari-vrp{N_CUSTOMERS}-real-c{CLUSTER_ID}"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# ------------------------------------------------------------------
# Load the zone train pool (real coords + fixed centroid depot) — same as H1
# ------------------------------------------------------------------
pool_path = Path(f"artifacts/instances/olist_sp_c{CLUSTER_ID}_train_pool.pt")
if not pool_path.exists():
    raise FileNotFoundError(f"{pool_path} not found. Run eda/05_split_train_test.py first.")
data  = torch.load(pool_path, map_location="cpu", weights_only=False)
pool  = data["coords"].numpy()      # (n_train, 2)
depot = data["depot"].numpy()       # (2,)

print(f"Run        : {RUN_NAME}  (baseline=POMO)")
print(f"Train pool : {len(pool):,} real customers (zone C{CLUSTER_ID})")
print(f"Depot      : ({depot[0]:.3f}, {depot[1]:.3f})  [fixed zone centroid]")
print(f"VRP{N_CUSTOMERS} | cap={VEHICLE_CAPACITY} | batch={BATCH_SIZE} × {N_CUSTOMERS} starts "
      f"= {BATCH_SIZE * N_CUSTOMERS} trajectories/step | epochs={N_EPOCHS}\n")

# ------------------------------------------------------------------
# Train
# ------------------------------------------------------------------
trainer = Trainer(
    n_customers=N_CUSTOMERS,
    vehicle_capacity=VEHICLE_CAPACITY,
    batch_size=BATCH_SIZE,
    embed_dim=128,
    lr=LR,
    max_grad_norm=MAX_GRAD_NORM,
    device=device,
    use_wandb=True,
    distribution="pool",
    pool=pool,
    depot=depot,
    baseline=BASELINE,
    run_name=RUN_NAME,
    wandb_project=WANDB_PROJECT,
)

print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}\n")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
trainer.train(
    n_epochs=N_EPOCHS,
    save_every=SAVE_EVERY,
    checkpoint_dir=CHECKPOINT_DIR,
)
print(f"Done. Checkpoints saved to {CHECKPOINT_DIR}\n")
