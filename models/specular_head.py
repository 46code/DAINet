"""Specular / clipped-pixel mask head (P1.3, optional).

Returns raw logits — the sigmoid is applied downstream (in
`models/network.py`, which exposes both `specular_logit` for the
numerically-safe `BCEWithLogitsLoss` and `specular_prob = sigmoid(logit)`
for any gating consumers). Folding the sigmoid into the loss avoids the
"unsafe to autocast" runtime error from `F.binary_cross_entropy` under
`torch.autocast`.

Zero-init of the final conv keeps the logit at 0 (prob 0.5) at init; the
BCE supervision pulls it toward 0 quickly so reconstruction losses are
not falsely down-weighted in the first epochs.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpecularHead(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, 1, kernel_size=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(feats))
        return self.conv2(x)
