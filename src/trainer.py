import copy
import os

import torch
import torch.nn as nn
from scipy import stats
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
    REINFORCE with pluggable baseline.

    baseline="none"   — vanilla REINFORCE (Williams 1992), no baseline
    baseline="ema"    — exponential moving average of past rewards (Nazari 2018)
    baseline="greedy" — greedy rollout baseline (Kool 2019)

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
        pool=None,
        depot=None,
        baseline="greedy",
        ema_alpha=0.99,
        run_name=None,
        wandb_project=None,
    ):
        self.n_customers = n_customers
        self.vehicle_capacity = vehicle_capacity
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.distribution = distribution
        self.pool = pool
        self.depot = depot
        self.use_wandb = use_wandb
        self.baseline = baseline
        self.ema_alpha = ema_alpha
        self.ema_value = 0.0  # running EMA state (used only when baseline="ema")
        self.run_name = run_name
        self.wandb_project = wandb_project

        self.actor = AttentionVRP(embed_dim=embed_dim).to(device)

        # Kool 2019: frozen copy of actor used as baseline, updated via t-test
        if baseline == "kool":
            self.actor_frozen = copy.deepcopy(self.actor)
            self.actor_frozen.eval()
            for p in self.actor_frozen.parameters():
                p.requires_grad_(False)
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
            pool=self.pool,
            depot=self.depot,
            device=self.device,
        )
        env = VRPEnvironment(coords, demands, vehicle_capacity=self.vehicle_capacity)

        with torch.amp.autocast("cuda", enabled=self.use_amp):
            log_probs_sum, rewards_sample = rollout(self.actor, env, greedy=False)

            if self.baseline in ("greedy", "kool"):
                baseline_actor = self.actor_frozen if self.baseline == "kool" else self.actor
                with torch.no_grad():
                    _, rewards_greedy = rollout(baseline_actor, env, greedy=True)
                advantage = rewards_sample - rewards_greedy
                baseline_value = rewards_greedy.mean().item()

            elif self.baseline == "ema":
                advantage = rewards_sample - self.ema_value
                baseline_value = self.ema_value
                # Update EMA after computing advantage
                self.ema_value = (self.ema_alpha * self.ema_value +
                                  (1 - self.ema_alpha) * rewards_sample.mean().item())

            else:  # "none" — vanilla REINFORCE
                advantage = rewards_sample
                baseline_value = 0.0

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
            "actor_loss":     actor_loss.item(),
            "critic_loss":    0.0,  # placeholder — non-zero when critic network is added
            "reward_mean":    rewards_sample.mean().item(),
            "reward_std":     rewards_sample.std().item(),
            "tour_length":    (-rewards_sample).mean().item(),
            "baseline_value": baseline_value,
            "advantage_mean": advantage.mean().item(),
            "advantage_std":  advantage.std().item(),
        }

    def _kool_try_update(self, n_eval=1000, significance=0.05):
        """
        Kool 2019: compare current actor vs frozen baseline on fresh instances.
        Replace frozen baseline if current is significantly better (paired t-test).
        Returns True if baseline was updated.
        """
        coords, demands = generate_batch(
            n_eval, self.n_customers,
            vehicle_capacity=self.vehicle_capacity,
            distribution=self.distribution,
            pool=self.pool,
            depot=self.depot,
            device=self.device,
        )
        env_cur  = VRPEnvironment(coords, demands, vehicle_capacity=self.vehicle_capacity)
        env_base = VRPEnvironment(coords, demands, vehicle_capacity=self.vehicle_capacity)

        with torch.no_grad():
            _, r_current = rollout(self.actor,        env_cur,  greedy=True)
            _, r_frozen  = rollout(self.actor_frozen, env_base, greedy=True)

        # Paired one-sided t-test: H1 = current is better (higher reward)
        t_stat, p_value = stats.ttest_rel(
            r_current.cpu().numpy(),
            r_frozen.cpu().numpy(),
        )
        updated = bool(t_stat > 0 and p_value / 2 < significance)
        if updated:
            self.actor_frozen = copy.deepcopy(self.actor)
            self.actor_frozen.eval()
            for p in self.actor_frozen.parameters():
                p.requires_grad_(False)
        return updated, p_value / 2

    def train(self, n_epochs, save_every=500, checkpoint_dir="artifacts/checkpoints",
              kool_eval_every=100, kool_n_eval=1000, kool_significance=0.05):
        if self.use_wandb:
            # VRP10 keeps the original project for backwards compatibility with existing runs.
            # Larger instances get their own project to keep dashboards clean.
            project = self.wandb_project or (
                "nazari-vrp-brazil" if self.n_customers == 10 else f"nazari-vrp{self.n_customers}"
            )
            wandb.init(
                project=project,
                name=self.run_name or f"vrp{self.n_customers}_{self.baseline}",
                config={
                    "n_customers": self.n_customers,
                    "batch_size": self.batch_size,
                    "distribution": self.distribution,
                    "device": self.device,
                    "baseline": self.baseline,
                },
            )

        for epoch in range(1, n_epochs + 1):
            metrics = self.train_step()

            # Kool: periodically test if frozen baseline should be updated
            if self.baseline == "kool" and epoch % kool_eval_every == 0:
                updated, p_val = self._kool_try_update(kool_n_eval, kool_significance)
                metrics["kool_baseline_updated"] = float(updated)
                metrics["kool_p_value"] = p_val
                if updated:
                    print(f"  [Kool] Epoch {epoch}: baseline updated (p={p_val:.4f})")

            if self.use_wandb:
                wandb.log({"epoch": epoch, **metrics})

            if epoch % 100 == 0:
                print(
                    f"Epoch {epoch:5d} | "
                    f"tour: {metrics['tour_length']:.4f} | "
                    f"adv: {metrics['advantage_mean']:.4f} ± {metrics['advantage_std']:.4f} | "
                    f"baseline: {metrics['baseline_value']:.4f} | "
                    f"loss: {metrics['actor_loss']:.4f}"
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
