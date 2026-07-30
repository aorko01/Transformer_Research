"""
Square-root blocked causal attention: local attention inside blocks, one global
attention *between* blocks, combined without ever building the full T x T map.

The sequence is cut into G blocks of L tokens each, with L = G = sqrt(T), so
for the usual T=1024 that is 32 blocks of 32 tokens. T must be a perfect
square.

  1. **Local.** Ordinary causal attention inside each block on its own:
     `local[g] = softmax(Q_g K_g^T / sqrt(head_dim))`, shape (G, L, L). Nothing
     crosses a block boundary here.
  2. **Summary.** Each local map is collapsed to one length-L vector by summing
     its rows together (`sum` over the *query* axis), i.e. "how much attention
     did each key position in this block receive". Stacking the G of them gives
     an (G, L) matrix - 32x32 in the running example.
     (Summing over the *key* axis instead would be useless: those rows are
     softmax outputs, so each one sums to exactly 1 and the summary would be a
     constant matrix of ones.)
  3. **Global.** That (G, L) matrix is treated as a sequence of G tokens with L
     features and self-attended causally: `glob = softmax(M M^T / sqrt(L))`,
     shape (G, G). No new projections - the summaries are their own Q and K.
  4. **Combine.** Output block i is built by scaling every local map by that
     block's row of the global map and applying the result to V:

         out[i] = sum_j glob[i, j] * (local[j] @ V[j])

     which is the `localscaled[j] = local[j] * glob[i][j]` construction, folded
     so the (G, G, L, L) stack of scaled maps is never a real tensor.

V is untouched until the very last step, exactly as in the vanilla path.

Memory
------
Only `local` (B, H, G, L, L) and `glob` (B, H, G, G) are kept for backward, so
the attention activations are B*H*T*sqrt(T) instead of vanilla's B*H*T*T - a 32x
reduction at T=1024. `_ScaledBlockCombine` is a hand-written autograd Function
precisely so the scaled maps stay virtual: forward contracts straight through
them, and backward reconstructs what it needs from `local` and `glob` on the
fly. Nothing of size (G, G, L, L) is ever allocated in either direction.

Causality
---------
Token (i, l) reads block j<i in full and its own block up to position l, so the
value path is strictly causal. Note that the *weights* are not: `glob[i, :]`
comes from the summary of block i, which sums over every query in that block,
so positions later in the same block influence how earlier ones mix the past.
The leak is bounded by the block (L-1 = 31 tokens at T=1024) and is inherent to
summarising a block before attending with it; shifting the global attention to
strictly past blocks (`j < i`) would remove it.
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from .attention_registry import BaseAttention, register_attention


def block_geometry(T):
    """
    (number of blocks, block length) for a sequence of T tokens.

    T must be a perfect square: L = G = sqrt(T). T=1024 is not a perfect
    square and would raise; T=1089 (33^2) gives a 33x33 split.
    """
    L = math.isqrt(max(T, 0))
    if L * L != T:
        raise ValueError(f"sequence length {T} must be a perfect square")
    G = L
    return G, L


class _ScaledBlockCombine(torch.autograd.Function):
    """
    out[b,h,i,l,:] = sum_j glob[b,h,i,j] * (local[b,h,j,l,:] @ V[b,h,j])

    Written by hand rather than left to autograd so that the intermediate
    per-output-block scaled maps (`local[j] * glob[i][j]`, a (G, G, L, L) object)
    are never materialised, and so the per-block products `local @ V` are
    recomputed in backward instead of being stashed by the forward pass.

    Shapes: local (B, H, G, L, L), glob (B, H, G, G), V (B, H, G, L, d).
    """

    @staticmethod
    def forward(ctx, local, glob, V):
        ctx.save_for_backward(local, glob, V)
        # P[j] = local[j] @ V[j]: apply each block's local map to its own values
        # once, then let the global map mix the blocks. Contracting in this order
        # is what keeps the scaled maps virtual.
        P = torch.matmul(local, V)
        B, H, G, L, d = P.shape
        out = torch.matmul(glob, P.reshape(B, H, G, L * d))
        return out.view(B, H, G, L, d)

    @staticmethod
    def backward(ctx, grad_out):
        local, glob, V = ctx.saved_tensors
        B, H, G, L, d = grad_out.shape
        grad_flat = grad_out.reshape(B, H, G, L * d)

        d_local = d_glob = d_V = None

        if ctx.needs_input_grad[1]:
            # rebuild the per-block outputs from what was saved rather than
            # having carried a (B, H, G, L, d) tensor through the forward
            P = torch.matmul(local, V).reshape(B, H, G, L * d)
            d_glob = torch.matmul(grad_flat, P.transpose(-2, -1))

        if ctx.needs_input_grad[0] or ctx.needs_input_grad[2]:
            # gradient w.r.t. P, i.e. the scaled maps summed back down per block
            dP = torch.matmul(glob.transpose(-2, -1), grad_flat).view(B, H, G, L, d)
            if ctx.needs_input_grad[0]:
                d_local = torch.matmul(dP, V.transpose(-2, -1))
            if ctx.needs_input_grad[2]:
                d_V = torch.matmul(local.transpose(-2, -1), dP)

        return d_local, d_glob, d_V


@register_attention("custom", "sqrt_block")
class CustomAttention(BaseAttention):
    """Drop-in replacement for `model.attention.Attention`; same in, same out."""

    def __init__(self, config):
        super().__init__(config)
        assert config.n_embd % config.n_head == 0
        self.Wq = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wk = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.Wv = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.block_size = config.block_size

        # One triangular mask serves both levels: the local maps slice [:L, :L]
        # and the global map slices [:G, :G]. Both G and L equal
        # sqrt(block_size), so this is 32x32 for a 1024-token context.
        G_max, L_max = block_geometry(config.block_size)
        span = max(G_max, L_max)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(span, span, dtype=torch.bool)),
            persistent=False,
        )

    def forward(self, X):
        Batch, Token, Embedding = X.shape
        assert Token <= self.block_size, (
            f"sequence length {Token} exceeds block_size {self.block_size}"
        )
        head_dim = Embedding // self.n_head
        G, L = block_geometry(Token)

        Q = self.Wq(X)
        K = self.Wk(X)
        V = self.Wv(X)

        # (Batch, Token, Embedding) -> (Batch, n_head, G, L, head_dim)
        def blocks(t):
            return t.view(Batch, G, L, self.n_head, head_dim).permute(0, 3, 1, 2, 4)

        Q, K, V = blocks(Q), blocks(K), blocks(V)

        # --- 1. local attention, each block against itself only ---------------
        #! might need to fix the scaling factor here
        scores = torch.matmul(Q, K.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        allowed = self.causal_mask[:L, :L]
        scores = scores.masked_fill(~allowed, float("-inf"))
        local = F.softmax(scores, dim=-1)

        # --- 2. one summary vector per block ---------------------------------
        # sum the rows of each local map together: entry m is the total
        # attention key m received from the queries in its block
        summary = local.sum(dim=-2)  # (Batch, n_head, G, L)

        # --- 3. attention between the blocks themselves -----------------------
        #! might need to fix the scaling factor here
        global_scores = torch.matmul(summary, summary.transpose(-2, -1)) * (1.0 / math.sqrt(L))
        global_scores = global_scores.masked_fill(~self.causal_mask[:G, :G], float("-inf"))
        glob = F.softmax(global_scores, dim=-1)  # (Batch, n_head, G, G)

        local = self.attn_dropout(local)
        glob = self.attn_dropout(glob)

        # Under autocast softmax returns fp32 while V is bf16; line the dtypes up
        # here so the hand-written backward (which runs outside autocast) sees a
        # single dtype, and so the big `local` tensor is saved at V's width.
        local = local.to(V.dtype)
        glob = glob.to(V.dtype)

        # --- 4. scale the local maps by the global row and apply to V ---------
        y = _ScaledBlockCombine.apply(local, glob, V)  # (Batch, n_head, G, L, head_dim)

        y = y.permute(0, 2, 3, 1, 4).reshape(Batch, G * L, Embedding)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y