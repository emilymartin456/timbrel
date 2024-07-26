import torch

from timbrel.config import ModelConfig
from timbrel.frontend import PhonemeVocab
from timbrel.models import build_model
from timbrel.prosody import transfer_prosody


def _cfg():
    return ModelConfig(
        vocab_size=len(PhonemeVocab()),
        hidden=16,
        n_enc_layers=2,
        n_dec_layers=2,
        n_heads=2,
        ffn_inner=32,
        ffn_kernel=3,
        speaker_dim=16,
        prosody_dim=16,
        n_speakers=4,
        variance_bins=16,
    )


def test_transfer_prosody_shape():
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    phonemes = torch.randint(1, cfg.vocab_size, (1, 5))
    src_lengths = torch.tensor([5])
    timbre = torch.randn(1, 12, 80)
    prosody = torch.randn(1, 20, 80)
    mel = transfer_prosody(model, phonemes, src_lengths, timbre, prosody)
    assert mel.shape[0] == 1 and mel.shape[2] == 80


def test_swapping_prosody_reference_changes_output():
    torch.manual_seed(0)
    cfg = _cfg()
    model = build_model(cfg)
    model.eval()
    phonemes = torch.randint(1, cfg.vocab_size, (1, 5))
    src_lengths = torch.tensor([5])
    timbre = torch.randn(1, 12, 80)
    mel_a = transfer_prosody(model, phonemes, src_lengths, timbre, torch.randn(1, 20, 80))
    mel_b = transfer_prosody(model, phonemes, src_lengths, timbre, torch.randn(1, 25, 80))
    # different prosody references should change the output (frames and/or values)
    if mel_a.shape == mel_b.shape:
        assert not torch.allclose(mel_a, mel_b)
