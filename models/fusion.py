"""Cross-attention fusion between feature-map tokens and an illumination embedding.

Queries are flattened feature tokens [B, H*W, C]; keys/values are projected from
the [B, E] illumination embedding to a single [B, 1, C] token. The attention
output is residually added to the feature tokens.

The output projection is zero-initialized so the fusion is the identity at
step 0 — combined with the decoder's zero-init R/L heads, the whole network is
the identity-on-RGB at init.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, feat_dim: int, embed_dim: int, num_heads: int = 8):
        super().__init__()
        # Round num_heads down to a divisor of feat_dim
        while feat_dim % num_heads != 0 and num_heads > 1:
            num_heads -= 1
        self.feat_norm = nn.LayerNorm(feat_dim)
        self.k_proj = nn.Linear(embed_dim, feat_dim)
        self.v_proj = nn.Linear(embed_dim, feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, num_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(feat_dim)
        # Zero-init the output projection -> identity at step 0
        nn.init.zeros_(self.attn.out_proj.weight)
        nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x: torch.Tensor, illum_emb: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B, H*W, C]
        q = self.feat_norm(tokens)
        k = self.k_proj(illum_emb).unsqueeze(1)  # [B, 1, C]
        v = self.v_proj(illum_emb).unsqueeze(1)  # [B, 1, C]
        attn_out, _ = self.attn(q, k, v)  # [B, H*W, C], zero at init
        out_tokens = tokens + self.out_norm(attn_out)
        return out_tokens.transpose(1, 2).reshape(B, C, H, W)
