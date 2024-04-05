"""Speaker-conditioned normalisation used to inject timbre into the decoder.

:class:`ConditionalLayerNorm` predicts the LayerNorm affine parameters from a
speaker embedding (the AdaSpeech trick): the bulk of the network stays
speaker-agnostic and identity is applied only through these lightweight
gain/bias terms, which makes them the natural target for few-shot adaptation.
"""

from __future__ import annotations

import torch
from torch import nn

from timbrel.models.layers import LinearNorm


class ConditionalLayerNorm(nn.Module):
    """LayerNorm whose scale/bias are produced from a conditioning vector."""

    def __init__(self, normalized_shape: int, cond_dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.scale = LinearNorm(cond_dim, normalized_shape)
        self.bias = LinearNorm(cond_dim, normalized_shape)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C); cond: (B, cond_dim)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        normed = (x - mean) / torch.sqrt(var + self.eps)
        gamma = self.scale(cond).unsqueeze(1)  # (B, 1, C), starts near 0
        beta = self.bias(cond).unsqueeze(1)
        return normed * (1.0 + gamma) + beta


class FiLM(nn.Module):
    """Feature-wise linear modulation from a conditioning vector.

    Unlike :class:`ConditionalLayerNorm` this does not normalise; it just
    applies an affine ``x * (1 + gamma) + beta`` where the parameters come from
    a conditioning vector (here, the prosody embedding).
    """

    def __init__(self, cond_dim: int, feature_dim: int) -> None:
        super().__init__()
        self.proj = LinearNorm(cond_dim, 2 * feature_dim)

    def affine_params(self, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gamma, beta = self.proj(cond).chunk(2, dim=-1)
        return gamma, beta

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.affine_params(cond)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
