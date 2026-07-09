import torch
from src.trainer import Trainer

device = "cpu"
print(f"Device: {device}\n")

trainer = Trainer(
    n_customers=10,
    vehicle_capacity=20,
    batch_size=4,
    embed_dim=128,
    lr=1e-4,
    device=device,
    use_wandb=False,
)

print(f"Actor params: {sum(p.numel() for p in trainer.actor.parameters()):,}\n")

trainer.train(n_epochs=3, save_every=999, checkpoint_dir="checkpoints_test")

print("\nSmoke test passed.")
