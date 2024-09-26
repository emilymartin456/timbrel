"""Reusable primitives: linear/conv with sane init, positional encoding."""

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
        if self.conv.bias is not None:
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
