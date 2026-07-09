import torch
from .model import AttentionVRP
from .utils import load_checkpoint


class RandomAgent:
    """
    Baseline agent that selects uniformly at random among valid (unmasked) nodes.

    Implements the same interface as AttentionVRP so it can be passed to rollout().
    No neural network — encode() and init_hidden() are no-ops.
    """

    def encode(self, static):
        return None

    def init_hidden(self, B, device):
        return None, None

    def step(self, static_emb, dynamic, last_node, hidden, cell, mask):
        # Uniform logits over valid nodes; -inf on masked nodes
        B, N = mask.shape
        logits = torch.zeros(B, N, device=mask.device)
        logits = logits.masked_fill(mask, float("-inf"))
        return logits, None, None


class NazariAgent:
    """
    Trained Nazari et al. (2018) agent.

    Wraps AttentionVRP and loads weights from a checkpoint file.
    Delegates encode(), init_hidden(), and step() directly to the model.
    """

    def __init__(self, checkpoint_path, embed_dim=128, device="cpu"):
        self.model = AttentionVRP(embed_dim=embed_dim).to(device)
        self.device = device
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()

    def encode(self, static):
        return self.model.encode(static)

    def init_hidden(self, B, device):
        return self.model.init_hidden(B, device)

    def step(self, static_emb, dynamic, last_node, hidden, cell, mask):
        with torch.no_grad():
            return self.model.step(static_emb, dynamic, last_node, hidden, cell, mask)
