import os
import torch
from src.trainer import Trainer

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Select VRP size ---
# Change this to 20 or 50
VRP_SIZE = 50

# --- Config per VRP size (Nazari 2018 capacities) ---
CONFIGS = {
    20: dict(vehicle_capacity=30, batch_size=512, n_epochs=20_000, save_every=2_000),
    50: dict(vehicle_capacity=40, batch_size=256, n_epochs=30_000, save_every=3_000),
}

cfg = CONFIGS[VRP_SIZE]

# --- Shared hyperparams ---
LR            = 1e-4
MAX_GRAD_NORM = 2.0
BASELINE      = "kool"

# VRP20: compare kool vs greedy; VRP50: compare embed_dim variants
RUNS = {
    20: [
        dict(embed_dim=128, run_name="vrp20_kool"),
        dict(embed_dim=128, run_name="vrp20_greedy", baseline="greedy"),
    ],
    50: [
        dict(embed_dim=128, run_name="vrp50_kool_emb128"),
        dict(embed_dim=64,  run_name="vrp50_kool_emb64"),
    ],
}

for run in RUNS[VRP_SIZE]:
    baseline  = run.get("baseline", BASELINE)
    embed_dim = run["embed_dim"]
    run_name  = run["run_name"]
    checkpoint_dir = f"checkpoints/{run_name}"

    print(f"{'='*50}")
    print(f"VRP{VRP_SIZE} | cap={cfg['vehicle_capacity']} | batch={cfg['batch_size']} | baseline={baseline} | embed={embed_dim} | epochs={cfg['n_epochs']}")

    trainer = Trainer(
        n_customers=VRP_SIZE,
        vehicle_capacity=cfg["vehicle_capacity"],
        batch_size=cfg["batch_size"],
        embed_dim=embed_dim,
        lr=LR,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
        use_wandb=True,
        distribution="uniform",
        baseline=baseline,
        run_name=run_name,
    )

    print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.train(
        n_epochs=cfg["n_epochs"],
        save_every=cfg["save_every"],
        checkpoint_dir=checkpoint_dir,
    )
    print(f"Done. Checkpoints saved to {checkpoint_dir}\n")
