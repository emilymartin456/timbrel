"""Timbrel — 双语多说话人声音克隆声学模型 (research prototype)."""

from timbrel.config import Config, ModelConfig, load_config, save_config
from timbrel.frontend.bilingual import BilingualFrontend
from timbrel.frontend.phonemes import PhonemeVocab
from timbrel.models import AcousticModel, build_model
from timbrel.version import __version__

__all__ = [
    "__version__",
    "Config",
    "ModelConfig",
    "load_config",
    "save_config",
    "BilingualFrontend",
    "PhonemeVocab",
    "AcousticModel",
    "build_model",
]
