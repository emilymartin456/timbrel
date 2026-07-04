"""Aggregate training objective for the Timbrel acoustic model.

Combines the reconstruction (mel L1), variance (duration/pitch/energy),
adversarial speaker-invariance and CLUB mutual-information terms, plus a small
regulariser on the FiLM parameters. Padding is respected everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from timbrel.config import TrainConfig


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    length = min(pred.size(1), target.size(1), pad_mask.size(1))
    pred, target, pad_mask = pred[:, :length], target[:, :length], pad_mask[:, :length]
    valid = (~pad_mask).unsqueeze(-1).to(pred.dtype)
    diff = F.l1_loss(pred, target, reduction="none") * valid
    return diff.sum() / valid.sum().clamp(min=1.0)


def _masked_mse_1d(pred: torch.Tensor, target: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    valid = (~pad_mask).to(pred.dtype)
    diff = ((pred - target) ** 2) * valid
    return diff.sum() / valid.sum().clamp(min=1.0)


class TimbrelLoss(nn.Module):
    """Weighted sum of every training term; returns ``(total, breakdown)``."""

    def __init__(self, config: TrainConfig | None = None) -> None:
        super().__init__()
        self.config = config or TrainConfig()

    def forward(self, outputs: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        c = self.config
        src_mask = outputs["src_mask"]
        mel_mask = outputs["mel_mask"]

        mel_target = targets["mel"]
        mel_loss = _masked_l1(outputs["mel"], mel_target, mel_mask)
        mel_loss = mel_loss + _masked_l1(outputs["postnet_mel"], mel_target, mel_mask)

        log_dur_target = torch.log(targets["durations"].float() + 1.0)
        duration_loss = _masked_mse_1d(outputs["log_duration"], log_dur_target, src_mask)
        pitch_loss = _masked_mse_1d(outputs["pitch"], targets["pitch"], src_mask)
        energy_loss = _masked_mse_1d(outputs["energy"], targets["energy"], src_mask)

        speaker_ids = targets["speaker_ids"]
        adv_loss = F.cross_entropy(outputs["content_speaker_logits"], speaker_ids)
        adv_loss = adv_loss + F.cross_entropy(outputs["prosody_speaker_logits"], speaker_ids)

        film_reg = (outputs["film_gamma"] ** 2).mean() + (outputs["film_beta"] ** 2).mean()
        club_mi = outputs.get("club_mi", torch.zeros((), device=mel_loss.device))

        total = (
            c.lambda_mel * mel_loss
            + c.lambda_duration * duration_loss
            + c.lambda_pitch * pitch_loss
            + c.lambda_energy * energy_loss
            + c.lambda_adv * adv_loss
            + c.lambda_club * club_mi
            + c.lambda_film_reg * film_reg
        )
        breakdown = {
            "total": total.detach(),
            "mel": mel_loss.detach(),
            "duration": duration_loss.detach(),
            "pitch": pitch_loss.detach(),
            "energy": energy_loss.detach(),
            "adversarial": adv_loss.detach(),
            "club_mi": club_mi.detach() if torch.is_tensor(club_mi) else club_mi,
            "film_reg": film_reg.detach(),
        }
        return total, breakdown
