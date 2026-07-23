"""Latent illumination CLS token (P2.2, optional).

A learned [CLS]-style token concatenated to the bottleneck sequence; one
multi-head cross-attention pass with the spatial tokens produces a global
illumination summary independent of (φ, θ) — usable at inference where the
direction is unknown.

The module returns an additive contribution to the illumination embedding;
the rest of the network treats this exactly like the `illum_emb` from
`IlluminationEmbedding`. At inference, this is the *only* source of
illumination conditioning (alongside the learned `null_illum_emb`).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LatentIlluminationToken(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.feat_proj = nn.Linear(feat_dim, embed_dim, bias=False)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            bias=True,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        # zero-init out_proj to preserve identity-at-init.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B, C, H, W] → returns [B, embed_dim] additive contribution."""
        B, C, H, W = feats.shape
        tokens = feats.flatten(2).transpose(1, 2)  # [B, HW, C]
        kv = self.feat_proj(tokens)  # [B, HW, embed_dim]
        q = self.cls.expand(B, -1, -1)  # [B, 1, embed_dim]
        attended, _ = self.attn(q, kv, kv, need_weights=False)  # [B, 1, embed_dim]
        return self.out_proj(attended.squeeze(1))
