"""Variance adaptor building blocks: predictors and the length regulator."""

from __future__ import annotations

import torch
from torch import nn

from timbrel.models.layers import ConvNorm, LinearNorm


class VariancePredictor(nn.Module):
    """2-layer conv predictor producing one scalar per input frame."""

    def __init__(
        self,
        in_dim: int,
        filter_size: int = 256,
        kernel_size: int = 3,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.conv_1 = ConvNorm(in_dim, filter_size, kernel_size, w_init_gain="relu")
        self.ln_1 = nn.LayerNorm(filter_size)
        self.conv_2 = ConvNorm(filter_size, filter_size, kernel_size, w_init_gain="relu")
        self.ln_2 = nn.LayerNorm(filter_size)
        self.dropout = nn.Dropout(dropout)
        self.proj = LinearNorm(filter_size, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = torch.relu(self.conv_1(x.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.ln_1(y))
        y = torch.relu(self.conv_2(y.transpose(1, 2))).transpose(1, 2)
        y = self.dropout(self.ln_2(y))
        out = self.proj(y).squeeze(-1)  # (B, T)
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
        return out


class DurationPredictor(VariancePredictor):
    """Predicts log-domain durations (frames per phoneme)."""


class LengthRegulator(nn.Module):
    """Expand phoneme-level features to frame level by repeating each step."""

    def forward(
        self,
        x: torch.Tensor,
        durations: torch.Tensor,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expanded: list[torch.Tensor] = []
        lengths: list[int] = []
        for seq, dur in zip(x, durations):
            # NOTE: naive per-step expansion; replaced with a vectorised path later
            pieces = []
            for vec, d in zip(seq, dur):
                d = int(d)
                if d > 0:
                    pieces.append(vec.unsqueeze(0).expand(d, -1))
            if pieces:
                repeated = torch.cat(pieces, dim=0)
            else:
                repeated = seq.new_zeros(0, seq.shape[-1])
            expanded.append(repeated)
            lengths.append(repeated.shape[0])

        target = max_len or max(lengths) or 1
        out = x.new_zeros(len(expanded), target, x.shape[-1])
        for i, seq in enumerate(expanded):
            out[i, : seq.shape[0]] = seq[:target]
        return out, torch.tensor(lengths, device=x.device, dtype=torch.long)
