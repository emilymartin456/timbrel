import torch

from timbrel.config import ModelConfig
from timbrel.frontend import PhonemeVocab
from timbrel.models import AcousticModel, build_model


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


def test_forward_shapes_with_ground_truth_durations():
    cfg = _tiny_config()
    model = AcousticModel(cfg)
    b, t, t_ref = 2, 5, 20
    phonemes = torch.randint(1, cfg.vocab_size, (b, t))
    src_lengths = torch.tensor([5, 4])
    ref_mel = torch.randn(b, t_ref, 80)
    durations = torch.tensor([[2, 2, 2, 2, 2], [3, 3, 2, 2, 0]])
    out = model(
        phonemes,
        src_lengths,
        ref_mel,
        durations=durations,
        pitch=torch.randn(b, t),
        energy=torch.randn(b, t),
    )
    expected_frames = int(max(durations[0].sum(), durations[1].sum()))
    assert out["postnet_mel"].shape == (b, expected_frames, 80)
    assert out["speaker"].shape == (b, cfg.speaker_dim)
    assert out["content_speaker_logits"].shape == (b, cfg.n_speakers)


def test_infer_runs_and_is_deterministic_in_eval():
    cfg = _tiny_config()
    model = build_model(cfg)
    model.eval()
    phonemes = torch.randint(1, cfg.vocab_size, (1, 6))
    src_lengths = torch.tensor([6])
    ref = torch.randn(1, 15, 80)
    mel_a, _ = model.infer(phonemes, src_lengths, ref)
    mel_b, _ = model.infer(phonemes, src_lengths, ref)
    assert mel_a.shape[0] == 1 and mel_a.shape[2] == 80
    assert torch.allclose(mel_a, mel_b)


def test_prosody_transfer_uses_separate_reference():
    cfg = _tiny_config()
    model = build_model(cfg)
    model.eval()
    phonemes = torch.randint(1, cfg.vocab_size, (1, 4))
    src_lengths = torch.tensor([4])
    timbre_ref = torch.randn(1, 15, 80)
    prosody_ref = torch.randn(1, 18, 80)
    mel, _ = model.infer(phonemes, src_lengths, timbre_ref, prosody_ref_mel=prosody_ref)
    assert mel.shape[0] == 1 and mel.shape[2] == 80
