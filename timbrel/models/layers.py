"""Reusable neural building blocks (linear/conv with sane init, positional enc)."""

from __future__ import annotations

import math

import torch
from torch import nn


class LinearNorm(nn.Module):
    """``nn.Linear`` with Xavier-uniform init on the weight."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear.weight, gain=nn.init.calculate_gain(w_init_gain))
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ConvNorm(nn.Module):
    """1-D convolution over ``(B, C, T)`` with Xavier init and 'same' padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int | None = None,
        dilation: int = 1,
        bias: bool = True,
        w_init_gain: str = "linear",
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to ``(B, T, D)`` inputs."""

    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


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

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
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

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.norm_1(x + self.dropout(self.attn(x, key_padding_mask)))
        x = self.norm_2(x + self.ffn(x))
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x
