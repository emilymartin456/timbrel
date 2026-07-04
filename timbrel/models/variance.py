"""Variance adaptor building blocks: predictors and the length regulator."""

from __future__ import annotations

import torch
from torch import nn

from timbrel.config import ModelConfig
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


class VarianceEmbedding(nn.Module):
    """Bucketise a scalar sequence into bins and embed it."""

    def __init__(self, n_bins: int, hidden: int, v_min: float, v_max: float) -> None:
        super().__init__()
        self.register_buffer("bins", torch.linspace(v_min, v_max, n_bins - 1))
        self.embed = nn.Embedding(n_bins, hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        idx = torch.bucketize(values, self.bins)
        return self.embed(idx)


class VarianceAdaptor(nn.Module):
    """Predict duration/pitch/energy and expand phoneme features to frames.

    Pitch and energy are predicted at the *phoneme* level (FastPitch-style),
    embedded through learned bins and added back before length regulation.
    Ground-truth values, when supplied, are used for teacher forcing.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        h = config.hidden
        self.duration_predictor = DurationPredictor(h)
        self.pitch_predictor = VariancePredictor(h)
        self.energy_predictor = VariancePredictor(h)
        self.pitch_embed = VarianceEmbedding(
            config.variance_bins, h, config.pitch_min, config.pitch_max
        )
        self.energy_embed = VarianceEmbedding(
            config.variance_bins, h, config.energy_min, config.energy_max
        )
        self.length_regulator = LengthRegulator()

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        durations: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        max_len: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        log_duration = self.duration_predictor(x, src_mask)
        pitch_pred = self.pitch_predictor(x, src_mask)
        energy_pred = self.energy_predictor(x, src_mask)

        pitch_target = pitch if pitch is not None else pitch_pred
        energy_target = energy if energy is not None else energy_pred
        x = x + self.pitch_embed(pitch_target) + self.energy_embed(energy_target)

        if durations is None:
            durations = torch.clamp(torch.round(torch.exp(log_duration) - 1.0), min=0).long()
        # zero out durations at padded phoneme positions so padding never
        # leaks frames into the regulated output (applies to GT durations too)
        durations = durations.masked_fill(src_mask, 0)

        expanded, mel_lengths = self.length_regulator(x, durations, max_len)
        stats = {
            "log_duration": log_duration,
            "pitch": pitch_pred,
            "energy": energy_pred,
            "durations": durations,
            "mel_lengths": mel_lengths,
        }
        return expanded, stats
