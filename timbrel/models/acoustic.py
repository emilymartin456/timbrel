"""Top-level Timbrel acoustic model.

This module starts as a thin skeleton; the encoder, speaker / prosody encoders,
variance adaptor and decoder are wired in over subsequent commits.
"""

from __future__ import annotations

from torch import nn

from timbrel.config import ModelConfig


class AcousticModel(nn.Module):
    """Non-autoregressive multi-speaker voice-cloning acoustic model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # TODO: build phoneme encoder / speaker & prosody encoders / variance / decoder

    def forward(self, *args, **kwargs):  # pragma: no cover - skeleton
        raise NotImplementedError("acoustic model is not assembled yet")
