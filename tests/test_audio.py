import numpy as np
import torch

from timbrel.audio.melbank import hz_to_mel, mel_filterbank, mel_to_hz
from timbrel.audio.pitch import average_by_duration, extract_f0, frame_energy
from timbrel.audio.stft import MelSpectrogram
from timbrel.config import AudioConfig


def test_mel_filterbank_shape_and_nonneg():
    fb = mel_filterbank(22050, 1024, 80)
    assert fb.shape == (80, 513)
    assert (fb >= 0).all()


def test_hz_mel_roundtrip():
    assert abs(float(mel_to_hz(hz_to_mel(440.0))) - 440.0) < 1e-3


def test_melspectrogram_shape():
    cfg = AudioConfig()
    mel_fn = MelSpectrogram(cfg)
    wav = torch.randn(1, cfg.sample_rate)
    mel = mel_fn(wav)
    assert mel.shape[0] == 1
    assert mel.shape[1] == cfg.n_mels
    expected = 1 + cfg.sample_rate // cfg.hop_length
    assert abs(mel.shape[2] - expected) <= 1


def test_extract_f0_recovers_sine_pitch():
    sr, freq = 22050, 220.0
    t = np.arange(sr) / sr
    wav = 0.5 * np.sin(2 * np.pi * freq * t)
    f0 = extract_f0(wav, sr, hop_length=256)
    voiced = f0[f0 > 0]
    assert voiced.size > 0
    assert abs(float(np.median(voiced)) - freq) < 20.0


def test_frame_energy_positive():
    log_mel = np.zeros((80, 10), dtype=np.float32)
    energy = frame_energy(log_mel)
    assert energy.shape == (10,)
    assert (energy > 0).all()


def test_average_by_duration():
    values = np.array([1.0, 1.0, 2.0, 2.0, 2.0], dtype=np.float32)
    out = average_by_duration(values, [2, 3])
    assert np.allclose(out, [1.0, 2.0])
