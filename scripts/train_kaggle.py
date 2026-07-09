import torch
from src.trainer import Trainer

# --- Device ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# --- Hyperparameters ---
# Capacities follow Nazari et al. (2018): VRP10→20, VRP20→30, VRP50→40
N_CUSTOMERS      = 20
VEHICLE_CAPACITY = 30
BATCH_SIZE       = 256
EMBED_DIM        = 128
LR               = 1e-4
MAX_GRAD_NORM    = 2.0
N_EPOCHS         = 20_000
SAVE_EVERY       = 1_000

trainer = Trainer(
    n_customers=N_CUSTOMERS,
    vehicle_capacity=VEHICLE_CAPACITY,
    batch_size=BATCH_SIZE,
    embed_dim=EMBED_DIM,
    lr=LR,
    max_grad_norm=MAX_GRAD_NORM,
    device=device,
    use_wandb=False,       # set True if wandb API key is configured on Kaggle
    distribution="uniform",
)

print(f"VRP{N_CUSTOMERS} | batch={BATCH_SIZE} | epochs={N_EPOCHS}")
print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}")
print()

trainer.train(n_epochs=N_EPOCHS, save_every=SAVE_EVERY, checkpoint_dir="checkpoints")
