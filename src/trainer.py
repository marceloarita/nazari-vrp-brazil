import os

import torch
import torch.nn as nn
from torch.distributions import Categorical

import wandb

from .environment import VRPEnvironment, generate_batch
from .model import AttentionVRP
from .utils import save_checkpoint


def rollout(actor, env, greedy=False):
    """
    Run a full episode batch using the actor policy.

    Args:
        actor:  AttentionVRP model
        env:    VRPEnvironment (already constructed, not yet reset)
        greedy: if True, take argmax instead of sampling

    Returns:
        log_probs_sum: (B,) — sum of log π(a_t|s_t) over each trajectory
        rewards:       (B,) — final reward (-total_distance) for each episode
    """
    static, dynamic, mask = env.reset()
    B = static.size(0)
    device = static.device

    static_emb = actor.encode(static)
    h, c = actor.init_hidden(B, device)
    current_node = torch.zeros(B, dtype=torch.long, device=device)

    done = torch.zeros(B, dtype=torch.bool, device=device)
    log_prob_steps = []

    # Upper bound: n_customers visits + at most n_customers depot returns
    max_steps = static.size(1) * 2

    for _ in range(max_steps):
        log_probs, h, c = actor.step(static_emb, dynamic, current_node, h, c, mask)

        if greedy:
            actions = log_probs.argmax(dim=-1)
        else:
            actions = Categorical(logits=log_probs).sample()

        # Each episode accumulates log-probs until its own termination; steps after done are zeroed
        step_log_p = log_probs[torch.arange(B, device=device), actions]
        step_log_p = step_log_p * (~done).float()
        log_prob_steps.append(step_log_p)

        (_, dynamic, mask), _, step_done = env.step(actions)
        done = done | step_done
        current_node = actions

        if done.all():
            break

    log_probs_sum = torch.stack(log_prob_steps, dim=1).sum(dim=1)  # (B,)
    return log_probs_sum, env.final_reward


class Trainer:
    """
    REINFORCE with greedy rollout baseline (Kool et al. 2019).

    NOTE: This departs from Nazari et al. (2018), which uses an exponential moving
    average of rewards as baseline. Kool's greedy rollout baseline is more stable
    and does not require a separate critic network.

    Each training step runs two rollouts over the same batch of instances:
      1. Sampling rollout  — actions drawn from π(·|s); used to compute the loss
      2. Greedy rollout    — actions = argmax π(·|s); used as the baseline

    Actor update:
        advantage = R_sample - R_greedy
        loss = -(advantage * log_probs_sum).mean()

    GPU optimizations:
      - Data generated directly on device (no CPU→GPU transfer for uniform distribution)
      - AMP (Automatic Mixed Precision) enabled on CUDA for ~2x throughput
    """

    def __init__(
        self,
        n_customers=50,
        vehicle_capacity=20,
        batch_size=128,
        embed_dim=128,
        lr=1e-4,
        max_grad_norm=2.0,
        device="cpu",
        use_wandb=True,
        distribution="uniform",
        kde=None,
    ):
        self.n_customers = n_customers
        self.vehicle_capacity = vehicle_capacity
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.distribution = distribution
        self.kde = kde
        self.use_wandb = use_wandb

        self.actor = AttentionVRP(embed_dim=embed_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        # AMP: enabled only on CUDA
        self.use_amp = device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

    def train_step(self):
        """Run one training iteration over a fresh batch of instances."""
        coords, demands = generate_batch(
            self.batch_size,
            self.n_customers,
            vehicle_capacity=self.vehicle_capacity,
            distribution=self.distribution,
            kde=self.kde,
            device=self.device,
        )
        env = VRPEnvironment(coords, demands, vehicle_capacity=self.vehicle_capacity)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            # --- Sampling rollout (gradient flows through log_probs_sum) ---
            log_probs_sum, rewards_sample = rollout(self.actor, env, greedy=False)

            # --- Greedy rollout as baseline (no gradient needed) ---
            with torch.no_grad():
                _, rewards_greedy = rollout(self.actor, env, greedy=True)

            # --- Actor loss: REINFORCE with greedy baseline (Kool 2019) ---
            advantage = rewards_sample - rewards_greedy
            actor_loss = -(advantage * log_probs_sum).mean()

        self.actor_opt.zero_grad()
        if self.use_amp:
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_opt)
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.scaler.step(self.actor_opt)
            self.scaler.update()
        else:
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "reward_mean": rewards_sample.mean().item(),
            "reward_std": rewards_sample.std().item(),
            "reward_greedy_mean": rewards_greedy.mean().item(),
        }

    def train(self, n_epochs, save_every=500, checkpoint_dir="checkpoints"):
        if self.use_wandb:
            wandb.init(
                project="nazari-vrp-brazil",
                config={
                    "n_customers": self.n_customers,
                    "batch_size": self.batch_size,
                    "distribution": self.distribution,
                    "device": self.device,
                    "baseline": "greedy_rollout",
                },
            )

        for epoch in range(1, n_epochs + 1):
            metrics = self.train_step()

            if self.use_wandb:
                wandb.log({"epoch": epoch, **metrics})

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"reward: {metrics['reward_mean']:.4f} ± {metrics['reward_std']:.4f} | "
                    f"greedy: {metrics['reward_greedy_mean']:.4f} | "
                    f"actor_loss: {metrics['actor_loss']:.4f}"
                )

            if epoch % save_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(checkpoint_dir, f"epoch_{epoch:05d}.pt"),
                    self.actor,
                    None,
                    self.actor_opt,
                    None,
                    epoch,
                )

        if self.use_wandb:
            wandb.finish()
