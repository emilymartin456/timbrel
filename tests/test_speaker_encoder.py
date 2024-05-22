import torch

from timbrel.models.speaker_encoder import SpeakerEncoder


def test_speaker_embedding_shape_and_unit_norm():
    enc = SpeakerEncoder(n_mels=80, hidden=32, speaker_dim=16)
    mel = torch.randn(4, 20, 80)
    emb = enc(mel)
    assert emb.shape == (4, 16)
    assert torch.allclose(emb.norm(dim=-1), torch.ones(4), atol=1e-4)


def test_speaker_encoder_respects_mask():
    enc = SpeakerEncoder(80, 32, 16)
    mel = torch.randn(2, 10, 80)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, 5:] = True
    emb = enc(mel, mask)
    assert emb.shape == (2, 16)
