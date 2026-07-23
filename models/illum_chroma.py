"""Illumination chroma-field factorisation — L = L_intensity · L_chroma.

Motivation. The documented residual-failure mode is a *smooth local colour
cast*: a red sidelight tints one wall, and the network restores the wall's
intensity but not its chromaticity. Physically, illumination **colour**
(chromaticity) varies smoothly across a scene, whereas illumination
**intensity** can fall off sharply (cast shadows, grazing light). The
standard single-`L` decoder ties both together, so a smooth colour cast has
to be represented in the same high-frequency map that also carries sharp
intensity fall-off — and the chroma residue leaks through.

This module factors the illumination map into:

    L = L_intensity (the existing per-pixel tanh-exp map, sharp allowed)
        · L_chroma   (a deliberately LOW-FREQUENCY per-pixel chromaticity
                      field with unit per-pixel geometric mean — pure colour,
                      no brightness)

`L_chroma` is predicted at a downsampled resolution and bilinearly
upsampled, which *bakes in* spatial smoothness (no smoothness loss needed),
and is geometric-mean-normalised across channels so it can only re-tint, not
re-brighten. This gives the decoder a dedicated, smooth degree of freedom to
absorb a local colour cast off the reflectance.

Identity-at-init: the projection is zero-init ⇒ ``L_chroma ≡ 1`` ⇒ ``L``
unchanged ⇒ the whole network is still bit-exactly identity on ``input_rgb``
at step 0. Off by default (``model.use_illum_chroma_field``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationChromaField(nn.Module):
    def __init__(self, in_channels: int, low_res_factor: int = 8, scale: float = 1.5):
        super().__init__()
        self.factor = max(1, int(low_res_factor))
        self.scale = float(scale)
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(
        self, feat: torch.Tensor, out_hw: tuple[int, int]
    ) -> torch.Tensor:
        """Return a smooth, unit-geomean chroma field [B, 3, out_hw]."""
        b, c, h, w = feat.shape
        lh = max(1, h // self.factor)
        lw = max(1, w // self.factor)
        low = F.adaptive_avg_pool2d(feat, (lh, lw))
        c_raw = self.proj(low)  # [B, 3, lh, lw], zero at init
        field = torch.exp(c_raw.tanh() * self.scale)  # =1 at init
        field = F.interpolate(
            field, size=out_hw, mode="bilinear", align_corners=False
        )
        # Normalise to unit per-pixel geometric mean across the 3 channels so
        # the field changes colour only (geomean=1 ⇒ no net brightness shift);
        # intensity stays entirely in L_intensity.
        log_f = torch.log(field.clamp_min(1e-4))
        log_f = log_f - log_f.mean(dim=1, keepdim=True)
        return torch.exp(log_f)
