import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticEncoder(nn.Module):
    """
    Projects node coordinates (x, y) into a D-dimensional embedding.

    Uses a 1x1 convolution — equivalent to a shared Linear(2→D) applied independently
    to every node. Same weights for all nodes; no cross-node context.

    Input:  (B, N+1, 2)    — coordinates for each node
    Output: (B, D, N+1)    — embedding per node
    """

    def __init__(self, input_dim=2, embed_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, N+1, 2) → transpose → (B, 2, N+1) → conv → (B, D, N+1)
        return self.conv(x.transpose(1, 2))


class DynamicEncoder(nn.Module):
    """
    Projects dynamic state [remaining_demand, remaining_capacity] into a D-dimensional embedding.

    Same architecture as StaticEncoder — shared Linear(2→D) per node, recomputed every step
    because the dynamic state changes as the vehicle visits nodes.

    Input:  (B, N+1, 2)    — [demand, remaining_cap] for each node
    Output: (B, D, N+1)    — embedding per node
    """

    def __init__(self, input_dim=2, embed_dim=128):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, embed_dim, kernel_size=1)

    def forward(self, x):
        # x: (B, N+1, 2) → transpose → (B, 2, N+1) → conv → (B, D, N+1)
        return self.conv(x.transpose(1, 2))


class Attention(nn.Module):
    """
    Two-pass attention (glimpse mechanism) from Nazari et al. 2018.

    Pass 1 (glimpse): aggregates a context vector from node embeddings weighted by the
                      LSTM hidden state. Incorporates both static and dynamic information.
    Pass 2 (pointer): produces the final action log-probabilities using the context from
                      pass 1 as the query. Uses tanh clipping (C=10) for stability.
    """

    def __init__(self, embed_dim=128, clip_c=10):
        super().__init__()
        self.clip_c = clip_c

        # Glimpse pass
        self.W_glimpse_ref = nn.Conv1d(embed_dim, embed_dim, 1)
        self.W_glimpse_q = nn.Linear(embed_dim, embed_dim)
        self.v_glimpse = nn.Conv1d(embed_dim, 1, 1)

        # Pointer pass
        self.W_ptr_ref = nn.Conv1d(embed_dim, embed_dim, 1)
        self.W_ptr_q = nn.Linear(embed_dim, embed_dim)
        self.v_ptr = nn.Conv1d(embed_dim, 1, 1)

    def forward(self, static_emb, dynamic_emb, query, mask):
        """
        Args:
            static_emb:  (B, D, N) — pre-computed static embeddings
            dynamic_emb: (B, D, N) — current dynamic embeddings
            query:       (B, D)    — LSTM hidden state h_t
            mask:        (B, N)    — True = node is forbidden

        Returns:
            log_probs: (B, N)
        """
        # Combined node representation: static + dynamic captures both fixed and changing info
        node_emb = static_emb + dynamic_emb                               # (B, D, N)

        # Pass 1 — glimpse: compute context vector
        ref = self.W_glimpse_ref(node_emb)                                # (B, D, N)
        q = self.W_glimpse_q(query).unsqueeze(-1)                         # (B, D, 1)
        alignment = self.v_glimpse(torch.tanh(ref + q)).squeeze(1)       # (B, N)
        alignment = alignment.masked_fill(mask, float("-inf"))
        weights = F.softmax(alignment, dim=-1)                            # (B, N)
        context = (weights.unsqueeze(1) * static_emb).sum(-1)            # (B, D)

        # Pass 2 — pointer: score each node using context as query
        ref = self.W_ptr_ref(static_emb)                                  # (B, D, N)
        q = self.W_ptr_q(context).unsqueeze(-1)                          # (B, D, 1)
        logits = self.v_ptr(torch.tanh(ref + q)).squeeze(1)              # (B, N)
        logits = self.clip_c * torch.tanh(logits)
        logits = logits.masked_fill(mask, float("-inf"))

        return F.log_softmax(logits, dim=-1)                              # (B, N)

#########################
# Actor model
#########################
class AttentionVRP(nn.Module):
    """
    Nazari et al. (2018) attention model for CVRP.

    Architecture:
      - StaticEncoder:  1D conv on coordinates → D=128 embeddings (computed once per episode)
      - DynamicEncoder: 1D conv on [demand, capacity] → D=128 embeddings (recomputed each step)
      - LSTMCell:       1-layer, hidden=128, input = static embedding of last visited node
      - Attention:      glimpse mechanism (2-pass) over node embeddings

    Usage pattern:
        static_emb = model.encode(static)          # once per episode
        h, c = model.init_hidden(B, device)
        for each decoding step:
            log_probs, h, c = model.step(static_emb, dynamic, last_node, h, c, mask)
    """

    def __init__(self, embed_dim=128, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.static_encoder = StaticEncoder(2, embed_dim)
        self.dynamic_encoder = DynamicEncoder(2, embed_dim)
        self.lstm = nn.LSTMCell(embed_dim, embed_dim)
        self.attention = Attention(embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, static):
        """
        Encode static node features (coordinates). Call once per episode.

        Args:
            static: (B, N+1, 2) — node coordinates; index 0 = depot

        Returns:
            static_emb: (B, embed_dim, N+1)
        """
        return self.static_encoder(static)

    def init_hidden(self, B, device):
        """Returns zero initial (hidden, cell) LSTM state."""
        z = torch.zeros(B, self.embed_dim, device=device)
        return z, z.clone()

    def step(self, static_emb, dynamic, last_node, hidden, cell, mask):
        """
        Single decoding step.

        Args:
            static_emb: (B, D, N+1) — from encode()
            dynamic:    (B, N+1, 2) — current [demand, remaining_capacity]
            last_node:  (B,) long   — index of last visited node
            hidden:     (B, D)      — LSTM hidden state
            cell:       (B, D)      — LSTM cell state
            mask:       (B, N+1)    — True = node is forbidden

        Returns:
            log_probs: (B, N+1), hidden: (B, D), cell: (B, D)
        """
        B = static_emb.size(0)
        device = static_emb.device

        dynamic_emb = self.dynamic_encoder(dynamic)                       # (B, D, N+1)

        # LSTM input: static embedding of the previously visited node
        last_emb = static_emb[torch.arange(B, device=device), :, last_node]  # (B, D)
        last_emb = self.dropout(last_emb)

        hidden, cell = self.lstm(last_emb, (hidden, cell))
        log_probs = self.attention(static_emb, dynamic_emb, hidden, mask)

        return log_probs, hidden, cell
