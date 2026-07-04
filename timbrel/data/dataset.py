"""Dataset and collate for training the acoustic model.

Kept storage-agnostic: a :class:`SpeechDataset` wraps a list of pre-extracted
sample dicts (phonemes, mel, durations, pitch, energy, speaker id). Feature
extraction from raw audio lives offline in ``scripts/`` so the training path
stays fast and dependency-light.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from timbrel.utils import pad_1d, pad_2d

REQUIRED_KEYS = ("phonemes", "mel", "durations", "pitch", "energy", "speaker_id")


class SpeechDataset(Dataset):
    """Thin wrapper over a list of pre-extracted sample dicts."""

    def __init__(self, samples: Sequence[dict]) -> None:
        for i, sample in enumerate(samples):
            missing = [k for k in REQUIRED_KEYS if k not in sample]
            if missing:
                raise KeyError(f"sample {i} is missing keys: {missing}")
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def collate_fn(batch: Sequence[dict]) -> dict:
    """Pad a list of samples into batched, model-ready tensors."""
    phonemes = [torch.as_tensor(b["phonemes"], dtype=torch.long) for b in batch]
    mels = [torch.as_tensor(b["mel"], dtype=torch.float) for b in batch]
    durations = [torch.as_tensor(b["durations"], dtype=torch.long) for b in batch]
    pitch = [torch.as_tensor(b["pitch"], dtype=torch.float) for b in batch]
    energy = [torch.as_tensor(b["energy"], dtype=torch.float) for b in batch]

    return {
        "phonemes": pad_1d(phonemes, 0),
        "src_lengths": torch.tensor([p.shape[0] for p in phonemes], dtype=torch.long),
        "mel": pad_2d(mels, 0.0),
        "mel_lengths": torch.tensor([m.shape[0] for m in mels], dtype=torch.long),
        "durations": pad_1d(durations, 0),
        "pitch": pad_1d(pitch, 0),
        "energy": pad_1d(energy, 0),
        "speaker_ids": torch.tensor([int(b["speaker_id"]) for b in batch], dtype=torch.long),
    }
