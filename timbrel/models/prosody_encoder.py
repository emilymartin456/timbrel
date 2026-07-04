"""Reference prosody encoder.

Summarises a reference utterance into a single prosody embedding and a pair of
FiLM (feature-wise affine) parameters that later modulate the content stream.
Because the prosody vector is trained adversarially to be speaker-invariant, it
can be lifted from one speaker and applied to another for prosody transfer.
"""

from __future__ import annotations

import torch
from torch import nn

from timbrel.models.layers import ConvNorm, LinearNorm


class ProsodyEncoder(nn.Module):
    """Log-mel ``(B, T, n_mels)`` -> prosody embedding + FiLM ``(gamma, beta)``."""

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        prosody_dim: int,
        feature_dim: int,
        n_layers: int = 2,
        kernel_size: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        in_ch = n_mels
        for _ in range(n_layers):
            self.convs.append(ConvNorm(in_ch, hidden, kernel_size, w_init_gain="relu"))
            in_ch = hidden
        self.dropout = nn.Dropout(dropout)
        self.proj = LinearNorm(hidden, prosody_dim)
        self.to_film = LinearNorm(prosody_dim, 2 * feature_dim)

    def forward(
        self, ref_mel: torch.Tensor, ref_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = ref_mel.transpose(1, 2)  # (B, n_mels, T)
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x)))
        x = x.transpose(1, 2)  # (B, T, hidden)
        if ref_mask is not None:
            valid = (~ref_mask).unsqueeze(-1).to(x.dtype)
            pooled = (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)
        else:
            pooled = x.mean(1)
        prosody = self.proj(pooled)  # (B, prosody_dim)
        gamma, beta = self.to_film(prosody).chunk(2, dim=-1)
        return prosody, gamma, beta
