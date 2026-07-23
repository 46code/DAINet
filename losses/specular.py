"""Specular / clipped-pixel BCE supervision (P1.3, off by default).

Pairs with `models.specular_head.SpecularHead`. The head emits raw logits;
the supervision target is derived from the *input* image's clipping (max
channel > threshold). Uses `binary_cross_entropy_with_logits`, which is
autocast-safe — `binary_cross_entropy` on probabilities crashes under
`torch.autocast` on CUDA.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpecularBCELoss(nn.Module):
    def __init__(self, threshold: float = 0.97):
        super().__init__()
        self.threshold = float(threshold)

    def forward(
        self,
        specular_logit: torch.Tensor,
        input_rgb: torch.Tensor,
    ) -> torch.Tensor:
        target = (input_rgb.amax(dim=1, keepdim=True) > self.threshold).to(specular_logit.dtype)
        return F.binary_cross_entropy_with_logits(specular_logit, target)
