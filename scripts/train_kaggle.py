import os
import torch
from src.trainer import Trainer

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}\n")

# --- Config ---
# Focus on VRP10 first; extend to VRP20/VRP50 after validating convergence.
# Capacity follows Nazari et al. (2018): VRP10 → 20.
CONFIGS = [
    {"n_customers": 10, "vehicle_capacity": 20, "batch_size": 512},
]

EMBED_DIM     = 128
LR            = 1e-4
MAX_GRAD_NORM = 2.0
N_EPOCHS      = 20_000
SAVE_EVERY    = 1_000

for cfg in CONFIGS:
    n  = cfg["n_customers"]
    vc = cfg["vehicle_capacity"]
    bs = cfg["batch_size"]
    checkpoint_dir = f"checkpoints/vrp{n}_cap{vc}"

    print(f"{'='*50}")
    print(f"VRP{n} | capacity={vc} | batch={bs} | epochs={N_EPOCHS}")

    trainer = Trainer(
        n_customers=n,
        vehicle_capacity=vc,
        batch_size=bs,
        embed_dim=EMBED_DIM,
        lr=LR,
        max_grad_norm=MAX_GRAD_NORM,
        device=device,
        use_wandb=True,
        distribution="uniform",
    )

    print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}\n")

    os.makedirs(checkpoint_dir, exist_ok=True)
    trainer.train(n_epochs=N_EPOCHS, save_every=SAVE_EVERY, checkpoint_dir=checkpoint_dir)
    print(f"Done. Checkpoints saved to {checkpoint_dir}\n")
