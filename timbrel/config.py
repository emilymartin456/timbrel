"""Configuration dataclasses for Timbrel models, audio frontend and training.

The whole config is a small tree of plain dataclasses so it can be built in
code, serialised to YAML, and diffed easily in experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AudioConfig:
    """Mel-spectrogram / STFT settings shared by extraction and the model."""

    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int = 1024
    n_mels: int = 80
    fmin: float = 0.0
    fmax: float = 8000.0


@dataclass
class FrontendConfig:
    """Text front-end behaviour (language routing, punctuation, boundaries)."""

    language: str = "auto"  # auto | zh | en
    insert_pause_on_punct: bool = True
    add_bos_eos: bool = False


@dataclass
class ModelConfig:
    """Acoustic model hyper-parameters."""

    n_mels: int = 80
    hidden: int = 256
    n_enc_layers: int = 4
    n_dec_layers: int = 4
    n_heads: int = 2
    ffn_kernel: int = 9
    ffn_inner: int = 1024
    dropout: float = 0.1
    speaker_dim: int = 128
    prosody_dim: int = 128
    n_speakers: int = 64
    variance_bins: int = 256
    pitch_min: float = 0.0
    pitch_max: float = 800.0
    energy_min: float = 0.0
    energy_max: float = 100.0
    max_positions: int = 2048
    grl_alpha: float = 1.0
    use_club: bool = True
    vocab_size: int = 0  # filled in from the phoneme vocabulary at build time


@dataclass
class TrainConfig:
    """Optimisation schedule and loss weights."""

    lr: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.98])
    weight_decay: float = 0.0
    batch_size: int = 16
    grad_clip: float = 1.0
    max_steps: int = 200_000
    warmup_steps: int = 4000
    seed: int = 1234
    lambda_mel: float = 1.0
    lambda_duration: float = 1.0
    lambda_pitch: float = 0.1
    lambda_energy: float = 0.1
    lambda_adv: float = 0.02
    lambda_club: float = 0.01
    lambda_film_reg: float = 1e-3


@dataclass
class Config:
    """Top-level container gathering every sub-config."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        data = data or {}
        return cls(
            audio=AudioConfig(**(data.get("audio") or {})),
            frontend=FrontendConfig(**(data.get("frontend") or {})),
            model=ModelConfig(**(data.get("model") or {})),
            train=TrainConfig(**(data.get("train") or {})),
        )
