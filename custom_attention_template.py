"""
custom_attention_template.py
============================
Drop-in replacement for BertSelfAttention.

Your class must:
  1. Accept (config: BertConfig) as its only constructor arg.
  2. Implement forward() with the signature shown below.
  3. Return a tuple: (context_layer,) or (context_layer, attn_weights)
     depending on output_attentions.

To use:
    python train.py --custom_attention custom_attention_template.LinearAttention
"""

import math
import torch
import torch.nn as nn
from transformers import BertConfig


class LinearAttention(nn.Module):
    """
    Example: vanilla linear (dot-product) attention — same math as
    BertSelfAttention but written from scratch so you can clearly see
    every hook point.

    Replace the body of forward() with your own attention mechanism.
    Everything else in the BERT stack (BertSelfOutput, BertAttention,
    BertLayer, etc.) stays untouched.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )

        self.num_heads   = config.num_attention_heads
        self.head_dim    = config.hidden_size // config.num_attention_heads
        self.all_head_sz = self.num_heads * self.head_dim

        # Q / K / V projections
        self.query = nn.Linear(config.hidden_size, self.all_head_sz)
        self.key   = nn.Linear(config.hidden_size, self.all_head_sz)
        self.value = nn.Linear(config.hidden_size, self.all_head_sz)

        self.attn_drop = nn.Dropout(config.attention_probs_dropout_prob)
        self.scale     = math.sqrt(self.head_dim)

    # ── helpers ────────────────────────────────────────────────────────

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, H*D) → (B, heads, T, D)"""
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, heads, T, D) → (B, T, H*D)"""
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.all_head_sz)

    # ── forward ────────────────────────────────────────────────────────

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
        past_key_value=None,
        output_attentions: bool = False,
    ):
        """
        Parameters
        ----------
        hidden_states : (B, T, hidden_size)
        attention_mask : (B, 1, 1, T) additive mask (0 for real, -10000 for pad)
        output_attentions : if True, return attention weights as second element

        Returns
        -------
        (context_layer,)  or  (context_layer, attn_weights)
        """
        # ── Project ───────────────────────────────────────────────────
        Q = self._split_heads(self.query(hidden_states))  # (B, h, T, d)
        K = self._split_heads(self.key(hidden_states))
        V = self._split_heads(self.value(hidden_states))

        # ── Scaled dot-product attention ─────────────────────────────
        scores = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (B, h, T, T)

        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        context = torch.matmul(attn_weights, V)             # (B, h, T, d)
        context = self._merge_heads(context)                 # (B, T, H*d)

        outputs = (context,)
        if output_attentions:
            outputs = outputs + (attn_weights,)

        return outputs


# ─── Stub for YOUR attention ─────────────────────────────────────────────────

class MyCustomAttention(LinearAttention):
    """
    Subclass LinearAttention and override forward() with your mechanism.
    Constructor accepts (config) just like BertSelfAttention.

    Usage:
        python train.py --custom_attention custom_attention_template.MyCustomAttention
    """

    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                encoder_hidden_states=None, encoder_attention_mask=None,
                past_key_value=None, output_attentions=False):

        # ← ← ← YOUR ATTENTION CODE HERE ← ← ←
        # For now, delegates to the standard scaled dot-product above.
        return super().forward(
            hidden_states, attention_mask, head_mask,
            encoder_hidden_states, encoder_attention_mask,
            past_key_value, output_attentions,
        )
