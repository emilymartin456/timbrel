"""Self-attention and the feed-forward Transformer (FFT) block.

Split out of :mod:`timbrel.models.layers` so the attention machinery lives on
its own. Both the phoneme encoder and the mel decoder build on these.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from timbrel.models.layers import ConvNorm, LinearNorm


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head self-attention with a key padding mask."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = LinearNorm(d_model, d_model)
        self.k_proj = LinearNorm(d_model, d_model)
        self.v_proj = LinearNorm(d_model, d_model)
        self.out_proj = LinearNorm(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, _ = x.shape
        shape = (b, t, self.n_heads, self.d_head)
        q = self.q_proj(x).view(*shape).transpose(1, 2)  # (B, H, T, Dh)
        k = self.k_proj(x).view(*shape).transpose(1, 2)
        v = self.v_proj(x).view(*shape).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if key_padding_mask is not None:
            # mask padded *keys* only, so no query row is ever fully masked
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)  # (B, H, T, Dh)
        out = out.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out_proj(out)


class ConvFFN(nn.Module):
    """Position-wise conv feed-forward network used inside an FFT block."""

    def __init__(self, d_model: int, d_inner: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.w_1 = ConvNorm(d_model, d_inner, kernel_size, w_init_gain="relu")
        self.w_2 = ConvNorm(d_inner, d_model, kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)  # (B, D, T)
        y = self.w_2(torch.relu(self.w_1(y)))
        y = y.transpose(1, 2)  # (B, T, D)
        return self.dropout(y)


class FFTBlock(nn.Module):
    """Feed-forward Transformer block: self-attention + conv FFN (post-norm)."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_inner: int,
        kernel_size: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn = ConvFFN(d_model, d_inner, kernel_size, dropout)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm_1(x + self.dropout(self.attn(x, key_padding_mask)))
        x = self.norm_2(x + self.ffn(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x
