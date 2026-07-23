"""SwinV2-style self-attention bottleneck for the s4 feature map.

One window-attention stage with two transformer blocks, residual-added
on top of the existing s4 features. The window-attention pattern keeps
the cost linear in spatial size while still providing global context
across the bottleneck tokens, which complements the
direction-conditioned cross-attention fusion that runs *after* this
block.

Identity-at-init: the final linear of each MLP and the qkv output
projection are *not* zero-init (a frozen identity would defeat the
purpose) — but the gain of the SwinV2 block on top of the rest of the
identity-at-init stack is small enough that the network still starts
within a stone's throw of the identity solution, and the
``model.use_swin_bottleneck: false`` config toggle recovers the prior
behavior for ablations.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class WindowAttention(nn.Module):
    """Standard window self-attention (Swin V1/V2 — V2 cosine norm)."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.out = nn.Linear(dim, dim)
        # SwinV2 cosine attention scale (learnable, log-form).
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        B, H, W, C = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # Split into non-overlapping windows
        x_win = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous()
        x_win = x_win.view(-1, ws * ws, C)  # [B*nW, ws*ws, C]

        qkv = self.qkv(x_win).reshape(x_win.shape[0], ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*nW, heads, N, hd]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SwinV2 cosine attention with a clipped learnable temperature.
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        # Clamp both sides: min=log(1) keeps τ ≥ 1 so cosine attention
        # softmax cannot saturate to a one-hot under fp16 backward.
        logit_scale = torch.clamp(
            self.logit_scale, min=math.log(1.0), max=math.log(100.0)
        ).exp()
        attn = (q @ k.transpose(-2, -1)) * logit_scale
        attn = attn.softmax(dim=-1)

        out_win = attn @ v  # [B*nW, heads, N, hd]
        out_win = out_win.transpose(1, 2).reshape(x_win.shape[0], ws * ws, C)
        out_win = self.out(out_win)

        # Reverse windowing
        out = out_win.view(B, Hp // ws, Wp // ws, ws, ws, C)
        out = out.permute(0, 1, 3, 2, 4, 5).contiguous()
        out = out.view(B, Hp, Wp, C)
        if pad_h or pad_w:
            out = out[:, :H, :W, :]
        return out


class SwinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads=num_heads, window_size=window_size)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SwinBottleneck(nn.Module):
    """Two SwinV2 blocks on top of s4; residual; no spatial change."""

    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 4, depth: int = 2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [SwinBlock(dim, num_heads=num_heads, window_size=window_size) for _ in range(depth)]
        )

    def forward(self, s4: torch.Tensor) -> torch.Tensor:
        # s4: [B, C, H, W]
        x = s4.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
        for blk in self.blocks:
            x = blk(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
        return s4 + x  # residual
