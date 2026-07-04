"""Reference-encoder that maps a short utterance to a speaker (d-vector) embedding."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from timbrel.models.layers import ConvNorm, LinearNorm


class SpeakerEncoder(nn.Module):
    """Log-mel ``(B, T, n_mels)`` -> L2-normalised speaker embedding ``(B, D)``.

    A small 1-D conv stack followed by (mask-aware) temporal average pooling and
    a projection. The output is unit-normalised so embeddings live on a sphere,
    which makes cosine averaging over multiple enrolment clips well-behaved.
    """

    def __init__(
        self,
        n_mels: int,
        hidden: int,
        speaker_dim: int,
        n_layers: int = 3,
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
        self.proj = LinearNorm(hidden, speaker_dim)

    def forward(self, ref_mel: torch.Tensor, ref_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = ref_mel.transpose(1, 2)  # (B, n_mels, T)
        for conv in self.convs:
            x = self.dropout(torch.relu(conv(x)))
        if ref_mask is not None:
            valid = (~ref_mask).unsqueeze(1).to(x.dtype)  # (B, 1, T)
            pooled = (x * valid).sum(-1) / valid.sum(-1).clamp(min=1.0)
        else:
            pooled = x.mean(-1)
        emb = self.proj(pooled)
        return F.normalize(emb, dim=-1)
