"""Neural building blocks and the Timbrel acoustic model."""

from timbrel.config import ModelConfig
from timbrel.models.acoustic import AcousticModel
from timbrel.models.layers import ConvNorm, LinearNorm, PositionalEncoding


def build_model(config: ModelConfig) -> AcousticModel:
    """Instantiate an :class:`AcousticModel` from a :class:`ModelConfig`."""
    return AcousticModel(config)


__all__ = [
    "LinearNorm",
    "ConvNorm",
    "PositionalEncoding",
    "AcousticModel",
    "build_model",
]
