"""Prosody feature extraction: fundamental frequency (f0) and frame energy.

The f0 tracker is a plain time-domain autocorrelation estimator — good enough
for building phoneme-level prosody targets offline, and dependency-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def frame_energy(log_mel: np.ndarray, floor: float = 1e-5) -> np.ndarray:
    """L2 energy per frame from a ``(n_mels, T)`` log-mel spectrogram."""
    log_mel = np.asarray(log_mel, dtype=np.float64)
    if log_mel.size == 0:
        return np.zeros(log_mel.shape[-1] if log_mel.ndim else 0, dtype=np.float32)
    # clip to keep exp() from overflowing on loud frames, and floor the result
    # so silent frames stay at a small positive value instead of collapsing to 0
    linear = np.exp(np.clip(log_mel, -30.0, 20.0))
    energy = np.linalg.norm(linear, axis=0)
    return np.maximum(energy, floor).astype(np.float32)


def extract_f0(
    wav: np.ndarray,
    sample_rate: int,
    hop_length: int,
    frame_length: int | None = None,
    fmin: float = 50.0,
    fmax: float = 600.0,
    voicing_threshold: float = 0.3,
) -> np.ndarray:
    """Per-frame f0 (Hz), with 0.0 marking unvoiced frames."""
    wav = np.asarray(wav, dtype=np.float64)
    if frame_length is None:
        frame_length = 4 * hop_length
    n_frames = 1 + len(wav) // hop_length
    f0 = np.zeros(n_frames, dtype=np.float32)

    min_lag = max(1, int(sample_rate / fmax))
    max_lag = int(sample_rate / fmin)
    pad = frame_length // 2
    padded = np.pad(wav, (pad, pad))

    for i in range(n_frames):
        start = i * hop_length
        frame = padded[start : start + frame_length]
        if frame.shape[0] < frame_length:
            break
        frame = frame - frame.mean()
        # TODO: O(n^2) autocorrelation; switch to an FFT-based ACF for long clips
        corr = np.correlate(frame, frame, mode="full")[frame_length - 1 :]
        if corr[0] <= 1e-8:
            continue
        hi = min(max_lag, len(corr) - 1)
        window = corr[min_lag:hi]
        if window.size == 0:
            continue
        lag = min_lag + int(np.argmax(window))
        if corr[lag] > voicing_threshold * corr[0]:
            f0[i] = float(sample_rate) / lag
    return f0


def average_by_duration(
    values: np.ndarray, durations: Sequence[int], voiced_only: bool = False
) -> np.ndarray:
    """Collapse frame-level ``values`` to phoneme level using ``durations``."""
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(len(durations), dtype=np.float32)
    pos = 0
    for i, d in enumerate(durations):
        d = int(d)
        if d > 0:
            segment = values[pos : pos + d]
            if voiced_only:
                segment = segment[segment > 0]
            out[i] = float(segment.mean()) if segment.size else 0.0
        pos += d
    return out
