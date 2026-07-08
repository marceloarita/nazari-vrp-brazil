import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

import wandb

from environment import VRPEnvironment, generate_batch
from model import AttentionVRP, StaticEncoder
from utils import save_checkpoint


class CriticNetwork(nn.Module):
    """
    Estimates V(X₀) — expected reward for the whole instance.

    Acts as the REINFORCE baseline. Reduces gradient variance without introducing bias.
    Takes the initial static instance as input (mean-pooled encoded features → MLP → scalar).
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        self.encoder = StaticEncoder(2, embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, static):
        # static: (B, N+1, 2)
        emb = self.encoder(static)           # (B, D, N+1)
        pooled = emb.mean(dim=-1)            # (B, D) — mean pooling over nodes
        return self.mlp(pooled).squeeze(-1)  # (B,)


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

        # Mask contribution of already-finished episodes (don't accumulate log-prob after done)
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
    Manages the Actor-Critic REINFORCE training loop for the Nazari VRP model.

    Actor update:  dθ = (1/N) Σ (R^n - V(X₀^n)) ∇θ log P(Y^n | X₀^n)
    Critic update: dφ = (1/N) Σ ∇φ (R^n - V(X₀^n))²

    Both networks share the same Adam optimizer with lr=1e-4 and gradient clipping at norm=2.
    """

    def __init__(
        self,
        n_customers=50,
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
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.distribution = distribution
        self.kde = kde
        self.use_wandb = use_wandb

        self.actor = AttentionVRP(embed_dim=embed_dim).to(device)
        self.critic = CriticNetwork(embed_dim=embed_dim).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

    def train_step(self):
        """Run one training iteration over a fresh batch of instances."""
        coords, demands = generate_batch(
            self.batch_size,
            self.n_customers,
            distribution=self.distribution,
            kde=self.kde,
            device=self.device,
        )
        env = VRPEnvironment(coords, demands)

        # --- Actor rollout ---
        log_probs_sum, rewards = rollout(self.actor, env)

        # --- Critic baseline (detached — does not flow gradient into actor update) ---
        baseline = self.critic(coords).detach()

        # --- Actor loss: REINFORCE with baseline ---
        advantage = rewards - baseline
        actor_loss = -(advantage * log_probs_sum).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # --- Critic loss: MSE against observed reward ---
        v_pred = self.critic(coords)
        critic_loss = F.mse_loss(v_pred, rewards.detach())

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "reward_mean": rewards.mean().item(),
            "reward_std": rewards.std().item(),
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
                    f"actor_loss: {metrics['actor_loss']:.4f} | "
                    f"critic_loss: {metrics['critic_loss']:.4f}"
                )

            if epoch % save_every == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                save_checkpoint(
                    os.path.join(checkpoint_dir, f"epoch_{epoch:05d}.pt"),
                    self.actor,
                    self.critic,
                    self.actor_opt,
                    self.critic_opt,
                    epoch,
                )

        if self.use_wandb:
            wandb.finish()
