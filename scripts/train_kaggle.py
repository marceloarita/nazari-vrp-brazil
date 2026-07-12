import os
import torch
from src.trainer import Trainer

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Select VRP size ---
# Change this to 20 or 50
VRP_SIZE = 20

# --- Config per VRP size (Nazari 2018 capacities) ---
CONFIGS = {
    20: dict(vehicle_capacity=30, batch_size=512, n_epochs=20_000, save_every=2_000),
    50: dict(vehicle_capacity=40, batch_size=256, n_epochs=30_000, save_every=3_000),
}

cfg = CONFIGS[VRP_SIZE]

# --- Shared hyperparams ---
EMBED_DIM     = 128
LR            = 1e-4
MAX_GRAD_NORM = 2.0

# Run one baseline per Kaggle session to avoid timeout (each ~4-8h for VRP20, longer for VRP50)
BASELINES = ["kool", "greedy"]

for baseline in BASELINES:
    checkpoint_dir = f"checkpoints/vrp{VRP_SIZE}_cap{cfg['vehicle_capacity']}_{baseline}"

    print(f"{'='*50}")
    print(f"VRP{VRP_SIZE} | capacity={cfg['vehicle_capacity']} | batch={cfg['batch_size']} | baseline={baseline} | epochs={cfg['n_epochs']}")

    trainer = Trainer(
        n_customers=VRP_SIZE,
        vehicle_capacity=cfg["vehicle_capacity"],
        batch_size=cfg["batch_size"],
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
    trainer.train(
        n_epochs=cfg["n_epochs"],
        save_every=cfg["save_every"],
        checkpoint_dir=checkpoint_dir,
    )
    print(f"Done. Checkpoints saved to {checkpoint_dir}\n")
