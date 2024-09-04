"""Triangular mel filterbank and hz<->mel conversion (HTK formula)."""

from __future__ import annotations

import numpy as np

_EPS = 1e-8


def hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=np.float64) / 700.0)


def mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> np.ndarray:
    """Return a ``(n_mels, n_fft // 2 + 1)`` triangular mel filterbank matrix."""
    if fmax is None:
        fmax = sample_rate / 2.0
    n_freqs = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sample_rate / 2.0, n_freqs)

    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        lower = (fft_freqs - left) / (center - left + _EPS)
        upper = (right - fft_freqs) / (right - center + _EPS)
        fb[m - 1] = np.maximum(0.0, np.minimum(lower, upper))
    return fb
