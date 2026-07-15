import torch
from .model import AttentionVRP


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

    def __init__(self, checkpoint_path, embed_dim=128, device="cpu", static_dim=None):
        # Auto-detect the static feature dim from the checkpoint (2 = coords only,
        # 3 = coords + density) so eval works without passing a flag.
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state = ckpt["actor_state"]
        if static_dim is None:
            static_dim = state["static_encoder.conv.weight"].shape[1]
        self.model = AttentionVRP(embed_dim=embed_dim, static_dim=static_dim).to(device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.device = device

    def encode(self, static):
        return self.model.encode(static)

    def init_hidden(self, B, device):
        return self.model.init_hidden(B, device)

    def step(self, static_emb, dynamic, last_node, hidden, cell, mask):
        with torch.no_grad():
            return self.model.step(static_emb, dynamic, last_node, hidden, cell, mask)
