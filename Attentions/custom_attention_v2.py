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


@register_attention("custom", "hierarchical")
class CustomAttention(BaseAttention):
    """
    Causal two-level hierarchical attention (multi-head), adapted from a
    bidirectional hierarchical_attention_block for decoder self-attention.

    Level 1 (local): ordinary causal self-attention within each contiguous
    group of S tokens.
    Level 2 (global): each query position attends to the pooled summaries of
    *strictly earlier* groups only. The query side uses a causal cumulative
    summary (cumsum over the in-group position axis), so an early position
    never sees a summary contaminated by later positions in its own group.
    A group's own summary is excluded from the global softmax entirely
    (whole-group pooling is only causally safe for groups that are wholly
    in the past); the query's own-group contribution is instead its Level-1
    local output, added back directly at full weight outside the softmax.

    w_o is shared between the group-summary ("squish") projection and the
    final output projection, matching the original single-head design.
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
        self.w_o = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        G_max, S_max = group_geometry(config.block_size)
        span = max(G_max, S_max)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask_strict",
            torch.tril(torch.ones(span, span, dtype=torch.bool), diagonal=-1),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, D = X.shape
        assert Token <= self.block_size
        H, hd = self.n_head, self.head_dim
        G, S = group_geometry(Token)
        padding = G * S - Token

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

        # ---- Level 1: local causal self-attention within each group ----
        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / self.scale
        allowed = self.causal_mask[:S, :S]
        if padding:
            real = (torch.arange(G * S, device=X.device) < Token).view(G, S)
            allowed = allowed & real.unsqueeze(-2)
        scores = scores.masked_fill(~allowed, float("-inf"))
        a = F.softmax(scores, dim=-1)
        if padding:
            a = a * real.view(G, S, 1).to(a.dtype)
        a = self.attn_dropout(a)
        raw_local = torch.matmul(a, Vh)  # (B,H,G,S,hd)

        # ---- squish into group summaries, through the shared w_o ----
        g = raw_local.sum(dim=-2)             # (B,H,G,hd)   whole-group pool -> KEY side
        g_causal = raw_local.cumsum(dim=-2)   # (B,H,G,S,hd) causal pool      -> QUERY side

        g_flat = g.permute(0, 2, 1, 3).reshape(Batch, G, D)
        g_causal_flat = g_causal.permute(0, 2, 3, 1, 4).reshape(Batch, G, S, D)

        G_prime = self.w_o(g_flat)               # (B,G,D)
        G_prime_causal = self.w_o(g_causal_flat) # (B,G,S,D)

        # ---- Level 2: causal attention to strictly earlier groups only ----
        Qg = self.w_q(G_prime_causal).view(Batch, G, S, H, hd).permute(0, 3, 1, 2, 4)  # (B,H,G,S,hd)
        Kg = self.w_k(G_prime).view(Batch, G, H, hd).permute(0, 2, 1, 3)               # (B,H,G,hd)

        global_scores = torch.einsum('bhgsd,bhjd->bhgsj', Qg, Kg) / self.scale
        strict_allowed = self.causal_mask_strict[:G, :G]
        global_scores = global_scores.masked_fill(
            ~strict_allowed.view(1, 1, G, 1, G), float("-inf")
        )
        a_prime = F.softmax(global_scores, dim=-1)
        a_prime = torch.nan_to_num(a_prime, nan=0.0)  # group 0 has no earlier groups
        a_prime = self.attn_dropout(a_prime)

        # mix in strictly-earlier groups' local outputs at the same relative
        # position, then add this group's own local output back directly,
        # outside the softmax, at full weight (causal-safe "own block" term)
        mixed = torch.einsum('bhgsj,bhjsd->bhgsd', a_prime, raw_local)
        y = raw_local + mixed  # (B,H,G,S,hd)

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * S, D)
        if padding:
            y = y[:, :Token]
        y = self.w_o(y)
        y = self.resid_dropout(y)
        return y