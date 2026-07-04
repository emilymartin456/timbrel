"""Phoneme encoder producing a speaker-invariant content representation."""

from __future__ import annotations

import torch
from torch import nn

from timbrel.config import ModelConfig
from timbrel.models.layers import FFTBlock, PositionalEncoding


class TextEncoder(nn.Module):
    """Embed phoneme ids and run them through a stack of FFT blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.hidden, padding_idx=0)
        self.pos = PositionalEncoding(config.hidden, config.max_positions)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [
                FFTBlock(
                    config.hidden,
                    config.n_heads,
                    config.ffn_inner,
                    config.ffn_kernel,
                    config.dropout,
                )
                for _ in range(config.n_enc_layers)
            ]
        )
        # instance norm removes utterance-level statistics -> a step towards a
        # speaker-invariant content bottleneck (AdaIN-VC style)
        self.content_norm = nn.InstanceNorm1d(config.hidden, affine=False)

    def forward(
        self, tokens: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embed(tokens)
        x = self.dropout(self.pos(x))
        for layer in self.layers:
            x = layer(x, key_padding_mask)
        x = self.content_norm(x.transpose(1, 2)).transpose(1, 2)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x
