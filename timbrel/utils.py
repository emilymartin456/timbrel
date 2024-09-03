"""Small tensor / reproducibility helpers used across the package."""

from __future__ import annotations

import random
from collections.abc import Sequence

import numpy as np
import torch


def seed_everything(seed: int = 1234) -> None:
    """Seed python, numpy and torch RNGs for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - deliberate global seeding for reproducibility
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mask_from_lengths(
    lengths: torch.Tensor, max_len: int | None = None
) -> torch.Tensor:
    """Build a padding mask from a batch of sequence lengths.

    Returns a bool tensor of shape ``(B, max_len)`` that is ``True`` at
    **padding** positions (i.e. index >= length), matching the convention used
    by :class:`torch.nn.MultiheadAttention`'s ``key_padding_mask``.
    """
    if max_len is None:
        max_len = int(lengths.max().item())
    ids = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return ids >= lengths.unsqueeze(1)


def pad_1d(sequences: Sequence[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Right-pad a list of 1-D tensors into a ``(B, T)`` tensor."""
    max_len = max(int(s.shape[0]) for s in sequences)
    out = sequences[0].new_full((len(sequences), max_len), pad_value)
    for i, s in enumerate(sequences):
        out[i, : s.shape[0]] = s
    return out


def pad_2d(sequences: Sequence[torch.Tensor], pad_value: float = 0.0) -> torch.Tensor:
    """Right-pad a list of ``(T_i, C)`` tensors into a ``(B, T_max, C)`` tensor."""
    max_len = max(int(s.shape[0]) for s in sequences)
    channels = sequences[0].shape[1]
    out = sequences[0].new_full((len(sequences), max_len, channels), pad_value)
    for i, s in enumerate(sequences):
        out[i, : s.shape[0]] = s
    return out
