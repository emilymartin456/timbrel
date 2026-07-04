"""Prosody transfer: synthesise text in one speaker's timbre with another
utterance's prosody.

Because the prosody encoder is trained (adversarially) to be speaker-invariant,
its FiLM output can be lifted from a *prosody reference* while the speaker
embedding (timbre) comes from a different *timbre reference*.
"""

from __future__ import annotations

from typing import Optional

import torch

from timbrel.models.acoustic import AcousticModel


@torch.no_grad()
def transfer_prosody(
    model: AcousticModel,
    phonemes: torch.Tensor,
    src_lengths: torch.Tensor,
    timbre_ref_mel: torch.Tensor,
    prosody_ref_mel: torch.Tensor,
    timbre_ref_lengths: Optional[torch.Tensor] = None,
    prosody_ref_lengths: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a mel-spectrogram with ``timbre_ref``'s voice and ``prosody_ref``'s prosody."""
    mel, _ = model.infer(
        phonemes,
        src_lengths,
        speaker_ref_mel=timbre_ref_mel,
        prosody_ref_mel=prosody_ref_mel,
        speaker_ref_lengths=timbre_ref_lengths,
        prosody_ref_lengths=prosody_ref_lengths,
    )
    return mel
