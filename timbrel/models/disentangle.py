"""Timbre / speaker disentanglement components.

Two complementary mechanisms are provided:

* an adversarial :class:`SpeakerClassifier` fed through a
  :class:`GradientReversal` layer, which pushes the content and prosody
  representations to be *speaker-invariant* (domain-adversarial training);
* (added later) a CLUB estimator giving a differentiable upper bound on the
  mutual information between content and speaker embeddings.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from timbrel.models.layers import LinearNorm


class _GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor):  # type: ignore[override]
        return -ctx.alpha * grad_output, None


def gradient_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReversalFn.apply(x, alpha)


class GradientReversal(nn.Module):
    """Identity in the forward pass; negates (scales) gradients in the backward."""

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reverse(x, self.alpha)


class SpeakerClassifier(nn.Module):
    """Small MLP predicting speaker id from a pooled representation."""

    def __init__(self, in_dim: int, n_speakers: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            LinearNorm(in_dim, hidden, w_init_gain="relu"),
            nn.ReLU(),
            LinearNorm(hidden, n_speakers),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
