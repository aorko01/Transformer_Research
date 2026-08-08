import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention_registry import BaseAttention, register_attention


def group_geometry(T):
    """(G, S): S = ceil(sqrt(T)) tokens per group, G = ceil(T / S) groups."""
    S = math.isqrt(max(T - 1, 0)) + 1
    G = (T + S - 1) // S
    return G, S


@register_attention("custom", "hierarchical_naive")
class CustomAttentionNaive(BaseAttention):
    """
    NAIVE decoder port of hierarchical_attention_block.

    Deliberately mirrors the original bidirectional block's structure as
    closely as possible. The ONLY change from the original is adding a
    causal mask at each of the two places attention scores are computed
    (local S x S scores, global G x G scores). Everything else -- the
    single whole-group sum `g` reused for both query and key sides, the
    dense (self-inclusive) group softmax, dropout applied once at the end
    only, `out = mixed` with no separate own-group addition -- is kept
    exactly as in the original.

    This is intentionally NOT a correct causal block. `g = raw_local.sum
    (dim=-2)` pools every position of a group, including positions after
    the one doing the reading, and that pooled vector is used to build
    BOTH Q_global and K_global. Masking global_scores to j <= i cannot
    undo that: the leak is already baked into Q_global before the mask
    (or the softmax) ever runs. See the `hierarchical` (non-naive)
    implementation for the fix (per-position causal cumsum query, and
    excluding the group's own summary from the global softmax entirely).
    """

    def __init__(self, config):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.block_size = config.block_size
        self.scale = math.sqrt(self.head_dim)

        self.w_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.w_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # shared between the squish (g -> G_prime) and final output
        # projection, matching the original single-head design
        self.w_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        # single dropout, applied once at the very end -- matches the
        # original exactly (no dropout on attention weights `a` / `a_prime`)
        self.dropout = nn.Dropout(config.dropout)

        G_max, S_max = group_geometry(config.block_size)
        span = max(G_max, S_max)
        # causal, diagonal included (j <= i) -- used for BOTH local and
        # global scores, since the original never excludes self-attention
        # at either level
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, D = X.shape
        assert Token <= self.block_size
        H, hd = self.n_head, self.head_dim
        G, S = group_geometry(Token)
        padding = G * S - Token

        # structural padding so T can be reshaped into a G x S grid --
        # unrelated to the original's input-level padding-token mask,
        # just a shape requirement of the group geometry
        Q = self.w_q(X)
        K = self.w_k(X)
        V = self.w_v(X)
        if padding:
            Q = F.pad(Q, (0, 0, 0, padding))
            K = F.pad(K, (0, 0, 0, padding))
            V = F.pad(V, (0, 0, 0, padding))

        def to_heads(t):
            return t.view(Batch, G, S, H, hd).permute(0, 3, 1, 2, 4)  # (B,H,G,S,hd)

        Qh, Kh, Vh = to_heads(Q), to_heads(K), to_heads(V)

        # ---- Level 1: local attention within each group, causal ----
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale
        allowed = self.causal_mask[:S, :S]
        if padding:
            real = (torch.arange(G * S, device=X.device) < Token).view(G, S)
            allowed = allowed & real.unsqueeze(-2)
        scores = scores.masked_fill(~allowed, float("-inf"))
        a = F.softmax(scores, dim=-1)
        if padding:
            a = a * real.view(G, S, 1).to(a.dtype)
        raw_local = torch.matmul(a, Vh)  # (B,H,G,S,hd)

        # ---- squish: single whole-group sum, exactly as original ----
        g = raw_local.sum(dim=-2)  # (B,H,G,hd)
        g_flat = g.permute(0, 2, 1, 3).reshape(Batch, G, D)  # (B,G,D)
        G_prime = self.w_o(g_flat)  # (B,G,D)

        # ---- Level 2: global attention, causal, dense/self-inclusive ----
        Qg = self.w_q(G_prime).view(Batch, G, H, hd).permute(0, 2, 1, 3)  # (B,H,G,hd)
        Kg = self.w_k(G_prime).view(Batch, G, H, hd).permute(0, 2, 1, 3)  # (B,H,G,hd)

        global_scores = torch.einsum('bhid,bhjd->bhij', Qg, Kg) / self.scale  # (B,H,G,G)
        allowed_g = self.causal_mask[:G, :G]  # j <= i, self included, same as original's dense softmax
        global_scores = global_scores.masked_fill(~allowed_g.view(1, 1, G, G), float("-inf"))
        a_prime = F.softmax(global_scores, dim=-1)  # every row has at least j=i -> no NaN case

        # exactly as original: `out`/`mixed` IS the final combine, no
        # separate additive own-group term
        mixed = torch.einsum('bhij,bhjsd->bhisd', a_prime, raw_local)  # (B,H,G,S,hd)
        y = mixed

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * S, D)
        if padding:
            y = y[:, :Token]
        y = self.w_o(y)
        y = self.dropout(y)
        return y