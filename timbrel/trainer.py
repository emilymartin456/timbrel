"""A minimal single-device trainer for the acoustic model.

Handles the optimisation loop, gradient clipping and the CLUB estimator's
separate learning objective. Distributed training, mixed precision and
checkpointing schedules are left for later.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch

from timbrel.config import Config
from timbrel.losses import TimbrelLoss
from timbrel.models.acoustic import AcousticModel
from timbrel.utils import seed_everything


class Trainer:
    """Own the model, optimiser and loss; step over batches from a dataloader."""

    def __init__(self, model: AcousticModel, config: Config, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        tc = config.train
        self.criterion = TimbrelLoss(tc)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc.lr,
            betas=(tc.betas[0], tc.betas[1]),
            weight_decay=tc.weight_decay,
        )
        self.step = 0

    def _to_device(self, batch: dict) -> dict:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v) for k, v in batch.items()}

    def train_step(self, batch: dict) -> dict:
        batch = self._to_device(batch)
        outputs = self.model(
            phonemes=batch["phonemes"],
            src_lengths=batch["src_lengths"],
            ref_mel=batch["mel"],
            ref_lengths=batch["mel_lengths"],
            durations=batch["durations"],
            pitch=batch["pitch"],
            energy=batch["energy"],
        )
        targets = {
            "mel": batch["mel"],
            "durations": batch["durations"],
            "pitch": batch["pitch"],
            "energy": batch["energy"],
            "speaker_ids": batch["speaker_ids"],
        }
        loss, breakdown = self.criterion(outputs, targets)
        # the CLUB variational network is trained by its own likelihood term
        if "club_learning" in outputs:
            loss = loss + self.config.train.lambda_club * outputs["club_learning"]

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
        self.optimizer.step()
        self.step += 1
        return breakdown

    def fit(self, dataloader: Iterable[dict], max_steps: int | None = None) -> list[dict]:
        seed_everything(self.config.train.seed)
        max_steps = max_steps or self.config.train.max_steps
        self.model.train()
        history: list[dict] = []
        while self.step < max_steps:
            for batch in dataloader:
                history.append(self.train_step(batch))
                if self.step >= max_steps:
                    break
        return history
