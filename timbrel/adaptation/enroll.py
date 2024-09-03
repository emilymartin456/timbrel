"""Few-shot speaker enrolment.

Builds a single speaker embedding by averaging the d-vectors extracted from a
handful of reference utterances of the target speaker. The mean is re-normalised
so it stays on the unit sphere, consistent with the speaker encoder's output.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F

from timbrel.models.acoustic import AcousticModel


class SpeakerEnroller:
    """Average speaker embeddings over reference clips of one speaker."""

    def __init__(self, model: AcousticModel) -> None:
        self.model = model

    @torch.no_grad()
    def enroll(self, ref_mels: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(ref_mels) == 0:
            raise ValueError("need at least one reference utterance to enroll")
        self.model.eval()
        embeds = []
        for mel in ref_mels:
            if mel.dim() == 2:
                mel = mel.unsqueeze(0)
            embeds.append(self.model.speaker_encoder(mel))
        mean = torch.cat(embeds, dim=0).mean(dim=0, keepdim=True)
        return F.normalize(mean, dim=-1)


def enroll(model: AcousticModel, ref_mels: Sequence[torch.Tensor]) -> torch.Tensor:
    """Convenience wrapper around :meth:`SpeakerEnroller.enroll`."""
    return SpeakerEnroller(model).enroll(ref_mels)
