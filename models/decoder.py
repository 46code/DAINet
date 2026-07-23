"""Multi-scale FiLM decoder with dual heads for reflectance R and illumination L.

Smooth-bounded parametrization (no hard clamps in the forward path):
    R = sigmoid(logit(x_rgb) + r_residual)      ∈ (0, 1)
    L = exp(tanh(l_log_raw) · 2.5)              ∈ [exp(-2.5), exp(2.5)] ≈ [0.082, 12.2]
Both are smoothly differentiable everywhere — no dead-zoned gradients on
saturated pixels. The 149× dynamic range on L is sufficient for indoor
sRGB scenes (real range 20–100×; 8-bit sensor clips at ~70×).

Identity-at-init:
- FiLM (gamma, beta) projection is zero-init -> (1+0)*x + 0 = x.
- head_R final conv is zero-init -> r_residual = 0
  -> R = sigmoid(logit(x_rgb)) = x_rgb (logit and sigmoid are inverses on (0,1)).
- head_L final conv is zero-init -> l_log_raw = 0
  -> L = exp(tanh(0) · 2.5) = exp(0) = 1.
- I_out = (R * L) = x_rgb at step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLM(nn.Module):
    """Per-channel affine modulation: y = (1 + gamma) * x + beta.

    gamma, beta are produced by an MLP from a conditioning embedding. The MLP's
    final linear is zero-initialized so the module is identity at step 0.
    """

    def __init__(self, embed_dim: int, feat_dim: int):
        super().__init__()
        self.feat_dim = feat_dim
        self.proj = nn.Linear(embed_dim, 2 * feat_dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, illum_emb: torch.Tensor) -> torch.Tensor:
        params = self.proj(illum_emb)  # [B, 2C]
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.feat_dim, 1, 1)
        beta = beta.view(-1, self.feat_dim, 1, 1)
        return (1.0 + gamma) * x + beta


class DecoderBlock(nn.Module):
    """Bilinear upsample, optional skip concat, two GN+Conv stages, FiLM."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, embed_dim: int):
        super().__init__()
        self.skip_ch = skip_ch
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.fuse = nn.Conv2d(in_ch + skip_ch, out_ch, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.film = FiLM(embed_dim, out_ch)

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None, illum_emb: torch.Tensor
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.act(self.gn1(self.fuse(x)))
        x = self.act(self.gn2(self.conv2(x)))
        x = self.film(x, illum_emb)
        return x


class DualHeadDecoder(nn.Module):
    """5-stage upsample from s4 with skips s3, s2, s1; two heads R and L."""

    def __init__(
        self,
        encoder_channels: list[int],
        embed_dim: int,
        out_channels: int = 3,
        use_illum_chroma_field: bool = False,
    ):
        super().__init__()
        self.use_illum_chroma_field = bool(use_illum_chroma_field)
        s1, s2, s3, s4 = encoder_channels
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(s4, s3, 256, embed_dim),  # /32 -> /16, +s3
                DecoderBlock(256, s2, 128, embed_dim),  # /16 -> /8, +s2
                DecoderBlock(128, s1, 64, embed_dim),  # /8 -> /4, +s1
                DecoderBlock(64, 0, 32, embed_dim),  # /4 -> /2
                DecoderBlock(32, 0, 32, embed_dim),  # /2 -> /1
            ]
        )
        self.head_R = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        self.head_L = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )
        # Zero-init the final 1x1 of each head -> identity at init
        for head in (self.head_R, self.head_L):
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

        # Optional illumination chroma-field: L = L_intensity · L_chroma,
        # where L_chroma is a smooth, unit-geomean (colour-only) field. Zero-
        # init ⇒ field ≡ 1 at step 0, so identity-at-init still holds. See
        # models/illum_chroma.py.
        if self.use_illum_chroma_field:
            from .illum_chroma import IlluminationChromaField

            self.illum_chroma = IlluminationChromaField(in_channels=32)

    def forward(
        self,
        s4: torch.Tensor,
        skips: list[torch.Tensor],
        illum_emb: torch.Tensor,
        x_rgb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        s1, s2, s3 = skips
        skip_seq = [s3, s2, s1, None, None]
        x = s4
        for block, skip in zip(self.blocks, skip_seq):
            x = block(x, skip, illum_emb)
        if x.shape[-2:] != x_rgb.shape[-2:]:
            x = F.interpolate(x, size=x_rgb.shape[-2:], mode="bilinear", align_corners=False)
        r_residual = self.head_R(x)
        l_log_raw = self.head_L(x)
        # R via sigmoid-of-logit: smooth gradient everywhere in (0, 1).
        # At init r_residual = 0 ⇒ R = sigmoid(logit(x_rgb)) = x_rgb.
        # The 1e-4 clamp on logit input is well above bf16 subnormal floor.
        r_pre = torch.logit(x_rgb.clamp(1e-4, 1.0 - 1e-4)) + r_residual
        R = torch.sigmoid(r_pre)
        # L via tanh-bounded exponent: ∈ [exp(-2.5), exp(2.5)] ≈ [0.082, 12.2].
        # tanh has smooth gradient everywhere — unlike the prior hard clamp,
        # head_L never receives zero gradient on saturated pixels.
        L = torch.exp(l_log_raw.tanh() * 2.5)
        # Optional smooth colour-only chroma field re-tints L without changing
        # its brightness (field ≡ 1 at init). Targets the local colour-cast
        # failure mode without smearing material edges.
        if self.use_illum_chroma_field:
            field = self.illum_chroma(x, out_hw=L.shape[-2:])
            L = L * field
        return R, L
