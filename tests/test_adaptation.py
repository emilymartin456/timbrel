import torch

from timbrel.adaptation import SpeakerEnroller, conditional_ln_parameters, freeze_for_adaptation
from timbrel.config import ModelConfig
from timbrel.frontend import PhonemeVocab
from timbrel.models import build_model


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


def test_enrollment_averages_reference_embeddings():
    model = build_model(_tiny_config())
    enroller = SpeakerEnroller(model)
    refs = [torch.randn(12, 80), torch.randn(18, 80), torch.randn(9, 80)]
    emb = enroller.enroll(refs)
    assert emb.shape == (1, 16)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(1), atol=1e-4)


def test_enrollment_requires_references():
    model = build_model(_tiny_config())
    try:
        SpeakerEnroller(model).enroll([])
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on empty enrolment")


def test_freeze_leaves_only_conditional_ln_trainable():
    model = build_model(_tiny_config())
    trainable = freeze_for_adaptation(model)
    cln_params = list(conditional_ln_parameters(model))
    assert len(trainable) == len(cln_params)
    assert all(p.requires_grad for p in cln_params)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    assert 0 < n_trainable < n_total  # a small fraction of the network
