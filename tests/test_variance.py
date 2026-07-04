import torch

from timbrel.config import ModelConfig
from timbrel.models.variance import LengthRegulator, VarianceAdaptor, VariancePredictor


def test_length_regulator_expands():
    lr = LengthRegulator()
    x = torch.randn(2, 3, 8)
    durations = torch.tensor([[1, 2, 3], [2, 2, 2]])
    out, lengths = lr(x, durations)
    assert out.shape == (2, 6, 8)
    assert lengths.tolist() == [6, 6]


def test_length_regulator_handles_zero_durations():
    lr = LengthRegulator()
    x = torch.randn(1, 3, 4)
    out, lengths = lr(x, torch.tensor([[0, 0, 0]]))
    assert lengths.tolist() == [0]
    assert out.shape[1] >= 1  # never produces a zero-width tensor


def test_variance_predictor_shape():
    vp = VariancePredictor(16)
    out = vp(torch.randn(2, 5, 16))
    assert out.shape == (2, 5)


def test_variance_adaptor_forward_with_targets():
    cfg = ModelConfig(hidden=16, variance_bins=32)
    adaptor = VarianceAdaptor(cfg)
    x = torch.randn(2, 4, 16)
    src_mask = torch.zeros(2, 4, dtype=torch.bool)
    durations = torch.tensor([[1, 1, 1, 1], [2, 1, 1, 0]])
    expanded, stats = adaptor(x, src_mask, durations=durations)
    assert expanded.shape[0] == 2
    assert expanded.shape[2] == 16
    for key in ("log_duration", "pitch", "energy", "mel_lengths"):
        assert key in stats
    assert stats["mel_lengths"].tolist() == [4, 4]
