"""Primary reconstruction: L1 between corrected image and flat-lit target."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def recon_l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)
