"""MaterialHead — auxiliary per-pixel material classifier.

Reads the deepest ConvNeXt feature ``s4`` (post-SwinV2 bottleneck, pre
illumination cross-attention fusion) and predicts a per-pixel class map
over the MIT-MI material taxonomy. Training-only: the head is supervised
by ``losses.material.material_ce_loss`` against
``batch['material_seg']`` and is skipped at val / inference (the
trainer passes ``compute_material=False``).

Design notes (minimal):

- Tap point post-SwinV2: SwinV2 has already injected global context but
  the illumination cross-attention has not yet polluted features with
  direction-specific residuals, so the material decision is made on
  illumination-invariant features.
- Zero-init the final conv → at step 0 the head outputs uniform logits,
  the CE loss is finite but uninformative, and the gradient signal is
  the cross-entropy training the head from scratch. Identity-at-init of
  the rest of the network is preserved (the head's output is never fed
  back into the reflectance / illumination pathway).
- Param cost on ConvNeXt-Base (in_channels=1024, hidden=256, K≈36):
  GroupNorm(1, 1024) ≈ 2K + Conv2d(1024, 256, 1) = 262K + Conv2d(256, K, 1)
  ≈ 9K ≈ 0.27M params.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaterialHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden: int = 256):
        super().__init__()
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        self.in_channels = in_channels
        self.num_classes = num_classes
        # GroupNorm(1, C) is mathematically a per-spatial LayerNorm and is the
        # cheap way to LN a [B, C, h, w] tensor without permuting.
        self.norm = nn.GroupNorm(1, in_channels)
        self.proj = nn.Conv2d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.logits = nn.Conv2d(hidden, num_classes, kernel_size=1)
        # Zero-init final conv so the head produces uniform logits at step 0.
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)

    def forward(self, s4: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        x = self.norm(s4)
        x = self.proj(x)
        x = self.act(x)
        x = self.logits(x)
        return F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
