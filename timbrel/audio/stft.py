"""Log-mel spectrogram extraction on top of ``torch.stft``.

Kept dependency-light on purpose (torch + numpy, no librosa): the mel basis is
precomputed once and registered as a buffer so the module moves cleanly across
devices.
"""

from __future__ import annotations

import torch
from torch import nn

from timbrel.audio.melbank import mel_filterbank
from timbrel.config import AudioConfig


class MelSpectrogram(nn.Module):
    """Waveform ``(B, T)`` -> log-mel ``(B, n_mels, frames)``."""

    def __init__(self, config: AudioConfig | None = None) -> None:
        super().__init__()
        self.config = config or AudioConfig()
        fb = mel_filterbank(
            self.config.sample_rate,
            self.config.n_fft,
            self.config.n_mels,
            self.config.fmin,
            self.config.fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(fb))
        self.register_buffer("window", torch.hann_window(self.config.win_length))

    def forward(self, wav: torch.Tensor, log: bool = True) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        spec = torch.stft(
            wav,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window,
            center=True,
            return_complex=True,
        )
        magnitude = spec.abs()  # (B, n_freqs, frames)
        mel = torch.matmul(self.mel_basis, magnitude)  # (B, n_mels, frames)
        if log:
            mel = torch.log(torch.clamp(mel, min=1e-5))
        return mel
