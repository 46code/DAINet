"""Retinex consistency loss (P2.1, off by default).

The decoder already exposes R and L heads, but the public output is
``head_R + reflectance_residual`` and L is exp(log_res); they aren't tied
together. This term *requires* ``I_out ≈ R · L`` (linear-RGB), turning the
two heads into a strict factorization.

We use linear-RGB because reflectance · illumination is physically a
multiplication in linear light. sRGB gamma would distort the relationship.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .color_ops import srgb_to_linear


class RetinexConstraintLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        I_out: torch.Tensor,
        R: torch.Tensor,
        L: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        I_lin = srgb_to_linear(I_out.clamp(0.0, 1.0))
        # If R is in sRGB-ish space too, convert; dainet's head_R clamps to
        # [0, 1] so the convention matches the output.
        R_lin = srgb_to_linear(R.clamp(0.0, 1.0))
        # L is multiplicative (>=0); take as-is.
        product = (R_lin * L.clamp_min(0.0)).clamp(0.0, 4.0)
        diff = (product - I_lin).abs()

        if mask is not None:
            m = mask
            if m.dim() == 3:
                m = m.unsqueeze(0)
            m = m.to(I_out.dtype).to(I_out.device).expand_as(diff)
            denom = m.sum().clamp_min(1.0)
            return (diff * m).sum() / denom
        return diff.mean()
