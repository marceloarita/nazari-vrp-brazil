import os
import torch
from src.trainer import Trainer

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Config ---
# VRP10 only. Three baselines trained sequentially for comparison.
# Capacity follows Nazari et al. (2018): VRP10 → 20.
N_CUSTOMERS   = 10
VEHICLE_CAP   = 20
BATCH_SIZE    = 512
EMBED_DIM     = 128
LR            = 1e-4
MAX_GRAD_NORM = 2.0
N_EPOCHS      = 10_000
SAVE_EVERY    = 1_000

BASELINES = ["kool"]

for baseline in BASELINES:
    checkpoint_dir = f"checkpoints/vrp{N_CUSTOMERS}_cap{VEHICLE_CAP}_{baseline}"

    print(f"{'='*50}")
    print(f"VRP{N_CUSTOMERS} | capacity={VEHICLE_CAP} | batch={BATCH_SIZE} | baseline={baseline} | epochs={N_EPOCHS}")

    trainer = Trainer(
        n_customers=N_CUSTOMERS,
        vehicle_capacity=VEHICLE_CAP,
        batch_size=BATCH_SIZE,
        embed_dim=EMBED_DIM,
        lr=LR,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
        use_wandb=True,
        distribution="uniform",
        baseline=baseline,
    )

    print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.train(n_epochs=N_EPOCHS, save_every=SAVE_EVERY, checkpoint_dir=checkpoint_dir)
    print(f"Done. Checkpoints saved to {checkpoint_dir}\n")
