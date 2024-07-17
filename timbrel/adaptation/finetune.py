"""Parameter-efficient speaker adaptation (AdaSpeech-style).

Only the speaker-conditional LayerNorm parameters are left trainable, so a new
voice can be fitted from a few utterances by updating a few thousand parameters
instead of the whole network — the key to stable few-shot cloning.
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
from torch import nn

from timbrel.losses import TimbrelLoss
from timbrel.models.condln import ConditionalLayerNorm


def conditional_ln_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Yield the parameters of every :class:`ConditionalLayerNorm` in ``model``."""
    for module in model.modules():
        if isinstance(module, ConditionalLayerNorm):
            yield from module.parameters()


def freeze_for_adaptation(model: nn.Module) -> list[nn.Parameter]:
    """Freeze the whole model except conditional LayerNorm; return trainables."""
    for param in model.parameters():
        param.requires_grad_(False)
    trainable = []
    for param in conditional_ln_parameters(model):
        param.requires_grad_(True)
        trainable.append(param)
    return trainable


def adapt(
    model: nn.Module,
    batches: Iterable[tuple[dict, dict]],
    criterion: TimbrelLoss | None = None,
    steps: int = 50,
    lr: float = 1e-4,
) -> list[float]:
    """Fine-tune only the conditional-LN parameters on ``(inputs, targets)`` batches."""
    trainable = freeze_for_adaptation(model)
    if not trainable:
        raise ValueError("model exposes no conditional layer norm to adapt")
    criterion = criterion or TimbrelLoss()
    optimizer = torch.optim.Adam(trainable, lr=lr)
    batch_list = list(batches)
    if not batch_list:
        raise ValueError("no adaptation batches provided")

    model.train()
    history: list[float] = []
    for step in range(steps):
        inputs, targets = batch_list[step % len(batch_list)]
        outputs = model(**inputs)
        loss, _ = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    return history
