"""Top-level Timbrel acoustic model tying every component together.

Data flow (training):

    phonemes ─► TextEncoder ─► content (speaker-invariant)
    ref_mel  ─► SpeakerEncoder ─► speaker embedding (timbre)
    ref_mel  ─► ProsodyEncoder ─► prosody embedding + FiLM(gamma, beta)

    content ──FiLM(prosody)──► VarianceAdaptor ──► LengthRegulator ──► MelDecoder(speaker)

The disentanglement heads (adversarial speaker classifiers via a gradient
reversal layer, plus an optional CLUB MI bound) operate on the pooled content
and prosody representations.
"""

from __future__ import annotations

import math  # noqa: F401  (kept during bring-up; removed in a later cleanup)

import torch
from torch import nn

from timbrel.config import ModelConfig
from timbrel.models.decoder import MelDecoder
from timbrel.models.disentangle import CLUBEstimator, GradientReversal, SpeakerClassifier
from timbrel.models.prosody_encoder import ProsodyEncoder
from timbrel.models.speaker_encoder import SpeakerEncoder
from timbrel.models.text_encoder import TextEncoder
from timbrel.models.variance import VarianceAdaptor
from timbrel.utils import get_mask_from_lengths


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = (~mask).unsqueeze(-1).to(x.dtype)
    return (x * valid).sum(1) / valid.sum(1).clamp(min=1.0)


class AcousticModel(nn.Module):
    """Non-autoregressive multi-speaker voice-cloning acoustic model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = TextEncoder(config)
        self.speaker_encoder = SpeakerEncoder(config.n_mels, config.hidden, config.speaker_dim)
        self.prosody_encoder = ProsodyEncoder(
            config.n_mels, config.hidden, config.prosody_dim, config.hidden
        )
        self.variance = VarianceAdaptor(config)
        self.decoder = MelDecoder(config)

        self.grl = GradientReversal(config.grl_alpha)
        self.content_speaker_clf = SpeakerClassifier(config.hidden, config.n_speakers)
        self.prosody_speaker_clf = SpeakerClassifier(config.prosody_dim, config.n_speakers)
        self.club = CLUBEstimator(config.hidden, config.speaker_dim) if config.use_club else None
        self._debug = False  # bring-up switch; removed in a later cleanup

    def _condition(
        self, content: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, src_mask: torch.Tensor
    ) -> torch.Tensor:
        cond = content * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        return cond.masked_fill(src_mask.unsqueeze(-1), 0.0)

    def forward(
        self,
        phonemes: torch.Tensor,
        src_lengths: torch.Tensor,
        ref_mel: torch.Tensor,
        ref_lengths: torch.Tensor | None = None,
        durations: torch.Tensor | None = None,
        pitch: torch.Tensor | None = None,
        energy: torch.Tensor | None = None,
        max_mel_len: int | None = None,
    ) -> dict:
        src_mask = get_mask_from_lengths(src_lengths, phonemes.size(1))
        ref_mask = (
            get_mask_from_lengths(ref_lengths, ref_mel.size(1)) if ref_lengths is not None else None
        )

        content = self.encoder(phonemes, src_mask)
        speaker = self.speaker_encoder(ref_mel, ref_mask)
        prosody, gamma, beta = self.prosody_encoder(ref_mel, ref_mask)
        cond = self._condition(content, gamma, beta, src_mask)

        expanded, var_out = self.variance(cond, src_mask, durations, pitch, energy, max_mel_len)
        mel_mask = get_mask_from_lengths(var_out["mel_lengths"], expanded.size(1))
        mel, postnet_mel = self.decoder(expanded, speaker, mel_mask)

        content_pooled = _masked_mean(content, src_mask)
        # if self._debug:
        #     print("content", content.shape, "speaker", speaker.shape)
        out = {
            "mel": mel,
            "postnet_mel": postnet_mel,
            "mel_mask": mel_mask,
            "src_mask": src_mask,
            "speaker": speaker,
            "prosody": prosody,
            "film_gamma": gamma,
            "film_beta": beta,
            "content_pooled": content_pooled,
            "content_speaker_logits": self.content_speaker_clf(self.grl(content_pooled)),
            "prosody_speaker_logits": self.prosody_speaker_clf(self.grl(prosody)),
            **var_out,
        }
        if self.club is not None:
            out["club_mi"] = self.club(content_pooled, speaker)
            out["club_learning"] = self.club.learning_loss(content_pooled, speaker)
        return out

    @torch.no_grad()
    def infer(
        self,
        phonemes: torch.Tensor,
        src_lengths: torch.Tensor,
        speaker_ref_mel: torch.Tensor,
        prosody_ref_mel: torch.Tensor | None = None,
        speaker_ref_lengths: torch.Tensor | None = None,
        prosody_ref_lengths: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Predict a mel-spectrogram. If ``prosody_ref_mel`` is given, timbre and
        prosody are taken from different references (prosody transfer)."""
        self.eval()
        prosody_ref = prosody_ref_mel if prosody_ref_mel is not None else speaker_ref_mel
        if prosody_ref_mel is None:
            prosody_ref_lengths = speaker_ref_lengths

        src_mask = get_mask_from_lengths(src_lengths, phonemes.size(1))
        spk_mask = (
            get_mask_from_lengths(speaker_ref_lengths, speaker_ref_mel.size(1))
            if speaker_ref_lengths is not None
            else None
        )
        pro_mask = (
            get_mask_from_lengths(prosody_ref_lengths, prosody_ref.size(1))
            if prosody_ref_lengths is not None
            else None
        )

        content = self.encoder(phonemes, src_mask)
        speaker = self.speaker_encoder(speaker_ref_mel, spk_mask)
        prosody, gamma, beta = self.prosody_encoder(prosody_ref, pro_mask)
        cond = self._condition(content, gamma, beta, src_mask)
        expanded, var_out = self.variance(cond, src_mask)
        mel_mask = get_mask_from_lengths(var_out["mel_lengths"], expanded.size(1))
        _, postnet_mel = self.decoder(expanded, speaker, mel_mask)
        return postnet_mel, var_out
