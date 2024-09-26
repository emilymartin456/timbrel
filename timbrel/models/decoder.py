"""Mel decoder with speaker-conditional layer norm, plus a conv post-net."""

from __future__ import annotations

import torch
from torch import nn

from timbrel.config import ModelConfig
from timbrel.models.attention import ConvFFN, MultiHeadAttention
from timbrel.models.condln import ConditionalLayerNorm
from timbrel.models.layers import ConvNorm, LinearNorm, PositionalEncoding


class DecoderBlock(nn.Module):
    """FFT block whose two LayerNorms are conditioned on the speaker embedding."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_inner: int,
        kernel_size: int,
        speaker_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cln_1 = ConditionalLayerNorm(d_model, speaker_dim)
        self.ffn = ConvFFN(d_model, d_inner, kernel_size, dropout)
        self.cln_2 = ConditionalLayerNorm(d_model, speaker_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        speaker: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.cln_1(x + self.dropout(self.attn(x, key_padding_mask)), speaker)
        x = self.cln_2(x + self.ffn(x), speaker)
        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return x


class PostNet(nn.Module):
    """5-layer conv post-net refining the coarse mel prediction (residual)."""

    def __init__(
        self, n_mels: int, channels: int = 256, kernel_size: int = 5, n_layers: int = 5
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        for i in range(n_layers):
            in_ch = n_mels if i == 0 else channels
            out_ch = n_mels if i == n_layers - 1 else channels
            gain = "linear" if i == n_layers - 1 else "tanh"
            self.convs.append(
                nn.Sequential(
                    ConvNorm(in_ch, out_ch, kernel_size, w_init_gain=gain),
                    nn.Identity() if i == n_layers - 1 else nn.Tanh(),
                    nn.Dropout(0.1),
                )
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        y = mel.transpose(1, 2)  # (B, n_mels, T)
        for conv in self.convs:
            y = conv(y)
        return y.transpose(1, 2)


class MelDecoder(nn.Module):
    """Frame-level decoder mapping expanded features to a mel-spectrogram."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.pos = PositionalEncoding(config.hidden, config.max_positions)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    config.hidden,
                    config.n_heads,
                    config.ffn_inner,
                    config.ffn_kernel,
                    config.speaker_dim,
                    config.dropout,
                )
                for _ in range(config.n_dec_layers)
            ]
        )
        self.mel_linear = LinearNorm(config.hidden, config.n_mels)
        self.postnet = PostNet(config.n_mels, config.hidden)

    def forward(
        self,
        x: torch.Tensor,
        speaker: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, speaker, mask)
        mel = self.mel_linear(x)
        postnet_mel = mel + self.postnet(mel)
        if mask is not None:
            mel = mel.masked_fill(mask.unsqueeze(-1), 0.0)
            postnet_mel = postnet_mel.masked_fill(mask.unsqueeze(-1), 0.0)
        return mel, postnet_mel
