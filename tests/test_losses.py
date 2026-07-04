import torch

from timbrel.config import ModelConfig, TrainConfig
from timbrel.frontend import PhonemeVocab
from timbrel.losses import TimbrelLoss
from timbrel.models.acoustic import AcousticModel


def _tiny_config():
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


def test_loss_terms_are_finite_and_differentiable():
    cfg = _tiny_config()
    model = AcousticModel(cfg)
    criterion = TimbrelLoss(TrainConfig())

    b, t, t_ref = 2, 5, 20
    phonemes = torch.randint(1, cfg.vocab_size, (b, t))
    src_lengths = torch.tensor([5, 5])
    ref_mel = torch.randn(b, t_ref, 80)
    durations = torch.tensor([[2, 2, 2, 2, 2], [2, 2, 2, 2, 2]])
    pitch = torch.randn(b, t)
    energy = torch.randn(b, t)

    out = model(phonemes, src_lengths, ref_mel, durations=durations, pitch=pitch, energy=energy)
    mel_len = out["postnet_mel"].size(1)
    targets = {
        "mel": torch.randn(b, mel_len, 80),
        "durations": durations,
        "pitch": pitch,
        "energy": energy,
        "speaker_ids": torch.randint(0, cfg.n_speakers, (b,)),
    }
    total, breakdown = criterion(out, targets)
    assert torch.isfinite(total)
    for key in ("mel", "duration", "pitch", "energy", "adversarial", "film_reg"):
        assert torch.isfinite(breakdown[key])
    total.backward()  # gradient graph is intact
