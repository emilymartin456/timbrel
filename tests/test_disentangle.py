import torch

from timbrel.models.disentangle import (
    CLUBEstimator,
    SpeakerClassifier,
    gradient_reverse,
)


def test_gradient_reversal_is_forward_identity():
    x = torch.randn(3, 4)
    assert torch.allclose(gradient_reverse(x, 1.0), x)


def test_gradient_reversal_negates_gradient():
    x = torch.randn(3, 4, requires_grad=True)
    gradient_reverse(x, 2.0).sum().backward()
    assert torch.allclose(x.grad, -2.0 * torch.ones_like(x))


def test_speaker_classifier_shape():
    clf = SpeakerClassifier(8, 5)
    assert clf(torch.randn(4, 8)).shape == (4, 5)


def test_club_estimator_is_finite():
    club = CLUBEstimator(8, 6)
    content = torch.randn(5, 8)
    speaker = torch.randn(5, 6)
    assert torch.isfinite(club(content, speaker))
    assert torch.isfinite(club.learning_loss(content, speaker))
