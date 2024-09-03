"""High-level synthesis API.

:class:`Synthesizer` ties the text front-end, the acoustic model and (at load
time) a saved checkpoint together, exposing a one-call ``clone`` method that
maps text plus a reference mel to a predicted mel-spectrogram. A neural vocoder
(mel -> waveform) is intentionally out of scope for this stage of the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

from timbrel.config import Config
from timbrel.frontend.bilingual import BilingualFrontend
from timbrel.frontend.phonemes import PhonemeVocab
from timbrel.models.acoustic import AcousticModel

PathLike = Union[str, Path]


class Synthesizer:
    """Text (+ reference mel) -> mel-spectrogram."""

    def __init__(
        self,
        model: AcousticModel,
        frontend: BilingualFrontend | None = None,
        config: Config | None = None,
    ) -> None:
        self.model = model
        self.config = config or Config()
        self.frontend = frontend or BilingualFrontend(PhonemeVocab(), self.config.frontend)

    @staticmethod
    def _batchify(mel: torch.Tensor | None) -> torch.Tensor | None:
        if mel is None:
            return None
        return mel if mel.dim() == 3 else mel.unsqueeze(0)

    @torch.no_grad()
    def clone(
        self,
        text: str,
        ref_mel: torch.Tensor,
        language: str | None = None,
        prosody_ref_mel: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ids = self.frontend.encode(text, language)
        if not ids:
            raise ValueError("text produced no phonemes")
        phonemes = torch.tensor([ids], dtype=torch.long)
        src_lengths = torch.tensor([len(ids)], dtype=torch.long)
        mel, _ = self.model.infer(
            phonemes,
            src_lengths,
            self._batchify(ref_mel),
            prosody_ref_mel=self._batchify(prosody_ref_mel),
        )
        return mel.squeeze(0)

    @classmethod
    def from_checkpoint(cls, path: PathLike, map_location: str = "cpu") -> Synthesizer:
        ckpt = torch.load(path, map_location=map_location)
        config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
        model = AcousticModel(config.model)
        model.load_state_dict(ckpt["model"])
        model.eval()
        return cls(model, config=config)
