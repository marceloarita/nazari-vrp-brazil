import torch
from environment import generate_batch, VRPEnvironment

# --- Setup ---
B = 1
N = 10
CAPACITY = 20

coords, demands = generate_batch(B, N, vehicle_capacity=CAPACITY)
env = VRPEnvironment(coords, demands, vehicle_capacity=CAPACITY)
static, dynamic, mask = env.reset()

print(f"VRP{N} | capacity: {CAPACITY}")
print(f"Demands (normalized): {demands[0, 1:].numpy().round(2)}")
print()

# --- Random episode ---
done = torch.zeros(B, dtype=torch.bool)
step = 0

while not done.all():
    # Pick a random valid action for each instance
    valid_indices = (~mask[0]).nonzero(as_tuple=True)[0]
    action = valid_indices[torch.randint(len(valid_indices), (1,))]  # shape: (B,) = (1,)

    (static, dynamic, mask), reward, done = env.step(action)
    step += 1

    node = action.item()
    label = "depot" if node == 0 else f"client {node}"
    print(f"step {step:2d}: {label:12s} | cap: {env.remaining_cap[0]:.2f} | reward: {reward[0]:.4f}")

# --- Results ---
print(f"\nTotal steps   : {step}")
print(f"Total distance: {env.total_dist[0]:.4f}")
print(f"Final reward  : {reward[0]:.4f}")
print(f"Tour          : {[t[0].item() for t in env.tour]}")

# --- Plot ---
env.render(
    batch_idx=0,
    title=f"VRP{N} — Random Policy | dist: {env.total_dist[0]:.3f}"
)
