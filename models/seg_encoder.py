"""Segmentation encoder: per-scene segment-id map -> global embedding.

The output embedding modulates the ConvNeXt RGB encoder via per-stage FiLM.
Segmentation never enters the RGB pixel stream; it conditions how features
evolve. The encoder is small (~150K params) and trained end-to-end.

Input:
    seg: [B, 1, H, W] int32 segment ids (0..N-1 per scene).
         Normalized to roughly [0, 1] by `SEG_NORMALIZATION` before convs.

Output:
    seg_emb: [B, embed_dim] float32 conditioning vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn


SEG_NORMALIZATION = 100.0


class SegmentationEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, hidden: int = 64):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden // 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden // 2), hidden // 2),
            nn.GELU(),
        )
        self.body = nn.Sequential(
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden), hidden),
            nn.GELU(),
            nn.Conv2d(hidden, hidden * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(min(8, hidden * 2), hidden * 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(hidden * 2, embed_dim)

    def forward(self, seg: torch.Tensor) -> torch.Tensor:
        x = seg.float() / SEG_NORMALIZATION
        x = self.stem(x)
        x = self.body(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
