"""Exponential moving average (EMA) of model parameters.

EMA shadow weights typically generalise better than live weights for
image-restoration networks (Karras et al. 2017, Zhang et al. 2022).
The trainer keeps both versions: live weights take gradients, EMA
weights are used for validation, and ``model_best.pt`` is written from
whichever has the better val PSNR.

Usage:

    ema = EMAModel(model, decay=0.999)
    for step in train_loop:
        optimizer.step()
        ema.update(model)
    # at validation:
    with ema.average_parameters(model):
        val_metrics_ema = evaluate(model)
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class EMAModel:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1); got {decay}")
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for name, p in model.named_parameters():
            if name not in self.shadow:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        self.shadow = {k: v.detach().clone() for k, v in state.items()}

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Swap live params for EMA inside the context, restore on exit."""
        backup: dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if name in self.shadow:
                backup[name] = p.detach().clone()
                p.data.copy_(self.shadow[name].data)
        try:
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name])
