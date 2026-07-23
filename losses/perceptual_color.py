"""Perceptual-color losses: differentiable ΔE2000 and L1-in-Lab.

These align the training objective with the metric we report. The dainet test
suite reports ΔE2000 as the primary number, but until now training only saw
L1 / log-chromaticity / region losses in sRGB or linear-RGB. Optimising
against a perceptual color metric (ΔE2000) directly is the single biggest
lever for pulling the test ΔE2000 down toward the < 3 target.

Both losses honor an optional boolean / float mask `[B, 1, H, W]` (1 = use,
0 = skip), so probe pixels and clipped highlights can be excluded from the
gradient. Masking is mandatory at train time when probe_mask / highlight_mask
are available — see `losses/manager.py`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .color_ops import srgb_to_linear


# Sqrt floors for the ΔE2000 chain. The chroma sqrts (C1, C2, dHp,
# Cbar7) need a comparatively large floor so their local gradient is
# bounded — 1/(2·√_EPS_CHROMA) = 5 per pixel at _EPS_CHROMA=1e-2.
# The FINAL sqrt(dE_sq) is on a sum of squared differences and only
# touches zero when pred ≡ target; we keep its floor tiny so identical
# inputs read ΔE ≈ 0 (numerical correctness). The near-gray detach block
# at lines 109–118 prevents gradient explosion through that final sqrt
# on near-identical pixels.
_EPS = 1e-2
_EPS_DE_SQ = 1e-12


def _srgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] → CIE Lab (D65). Returns [B, 3, H, W]. Differentiable.

    Mirrors `evaluation.metrics._srgb_to_lab` but with `clamp_min(_EPS)` on
    every sqrt/pow input so gradients stay finite at the origin.
    """
    rgb = rgb.clamp(0.0, 1.0)
    lin = srgb_to_linear(rgb)
    M = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        device=rgb.device,
        dtype=rgb.dtype,
    )
    xyz = torch.einsum("ij,bjhw->bihw", M, lin)
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x_n = xyz[:, 0:1] / Xn
    y_n = xyz[:, 1:2] / Yn
    z_n = xyz[:, 2:3] / Zn
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0

    def f(t: torch.Tensor) -> torch.Tensor:
        t_safe = t.clamp_min(_EPS)
        return torch.where(
            t > epsilon,
            t_safe.pow(1.0 / 3.0),
            (kappa * t + 16.0) / 116.0,
        )

    fx, fy, fz = f(x_n), f(y_n), f(z_n)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.cat([L, a, b], dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    m = mask
    if m.dim() == 4 and m.shape[1] == 1:
        m = m.squeeze(1)
    elif m.dim() == 3 and m.shape[0] == 1 and values.dim() == 3:
        m = m.expand_as(values)
    m = m.to(values.dtype).to(values.device)
    if m.shape != values.shape:
        m = m.expand_as(values)
    denom = m.sum().clamp_min(1.0)
    return (values * m).sum() / denom


def delta_e2000_per_pixel_differentiable(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Per-pixel ΔE2000 with gradient-safe sqrts / atan2.

    Returns [B, H, W]. The non-differentiable corners of the Sharma 2000
    formula (sqrt at the origin, atan2 at (0, 0)) are guarded with `_EPS`.
    """
    lab1 = _srgb_to_lab(pred)
    lab2 = _srgb_to_lab(target)
    L1, a1, b1 = lab1[:, 0], lab1[:, 1], lab1[:, 2]
    L2, a2, b2 = lab2[:, 0], lab2[:, 1], lab2[:, 2]

    C1 = torch.sqrt((a1 ** 2 + b1 ** 2).clamp_min(_EPS))
    C2 = torch.sqrt((a2 ** 2 + b2 ** 2).clamp_min(_EPS))
    Cbar = (C1 + C2) / 2.0
    Cbar7 = Cbar.clamp_min(_EPS) ** 7
    G = 0.5 * (1.0 - torch.sqrt(Cbar7 / (Cbar7 + 25.0 ** 7)))
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    # atan2(b, a) at (a, b) = (0, 0) has NaN gradient (∂/∂a = -b/(a²+b²)),
    # so we must detach hue derivatives on near-gray pixels. The detach
    # threshold compares the *un-clamped* chroma magnitude squared
    # `a²+b²` against `1e-3` (chroma radius < ~0.032 in Lab) — using the
    # `clamp_min(_EPS)` version of `C1p` would skip the guard entirely
    # because the clamp floors `C1p` to sqrt(_EPS) ≈ 0.01 > 1e-3.
    # Use a near-gray threshold larger than the sqrt floor _EPS so the
    # detach actually fires before the clamp floors C1p / C2p.
    _NEAR_GRAY = max(_EPS * 10.0, 1e-1)
    raw_C1p_sq = (a1p ** 2 + b1 ** 2).detach()
    raw_C2p_sq = (a2p ** 2 + b2 ** 2).detach()
    near_gray_1 = raw_C1p_sq < _NEAR_GRAY
    near_gray_2 = raw_C2p_sq < _NEAR_GRAY
    C1p = torch.sqrt((a1p ** 2 + b1 ** 2).clamp_min(_EPS))
    C2p = torch.sqrt((a2p ** 2 + b2 ** 2).clamp_min(_EPS))
    a1p_safe = torch.where(near_gray_1, a1p.detach(), a1p)
    b1_safe = torch.where(near_gray_1, b1.detach(), b1)
    a2p_safe = torch.where(near_gray_2, a2p.detach(), a2p)
    b2_safe = torch.where(near_gray_2, b2.detach(), b2)
    h1p = torch.atan2(b1_safe, a1p_safe) % (2 * math.pi)
    h2p = torch.atan2(b2_safe, a2p_safe) % (2 * math.pi)

    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = h2p - h1p
    dhp = torch.where(dhp > math.pi, dhp - 2 * math.pi, dhp)
    dhp = torch.where(dhp < -math.pi, dhp + 2 * math.pi, dhp)
    dHp = 2.0 * torch.sqrt((C1p * C2p).clamp_min(_EPS)) * torch.sin(dhp / 2.0)

    Lbarp = (L1 + L2) / 2.0
    Cbarp = (C1p + C2p) / 2.0
    hbarp = (h1p + h2p) / 2.0
    hbarp = torch.where(torch.abs(h1p - h2p) > math.pi, hbarp + math.pi, hbarp)

    T = (
        1.0
        - 0.17 * torch.cos(hbarp - math.radians(30.0))
        + 0.24 * torch.cos(2 * hbarp)
        + 0.32 * torch.cos(3 * hbarp + math.radians(6.0))
        - 0.20 * torch.cos(4 * hbarp - math.radians(63.0))
    )
    dTheta = math.radians(30.0) * torch.exp(-(((hbarp * 180.0 / math.pi - 275.0) / 25.0) ** 2))
    Cbarp7 = Cbarp.clamp_min(_EPS) ** 7
    Rc = 2.0 * torch.sqrt(Cbarp7 / (Cbarp7 + 25.0 ** 7))
    Sl = 1.0 + (0.015 * (Lbarp - 50.0) ** 2) / torch.sqrt(20.0 + (Lbarp - 50.0) ** 2)
    Sc = 1.0 + 0.045 * Cbarp
    Sh = 1.0 + 0.015 * Cbarp * T
    Rt = -torch.sin(2.0 * dTheta) * Rc

    dE_sq = (
        (dLp / Sl) ** 2
        + (dCp / Sc) ** 2
        + (dHp / Sh) ** 2
        + Rt * (dCp / Sc) * (dHp / Sh)
    )
    return torch.sqrt(dE_sq.clamp_min(_EPS_DE_SQ))


class DeltaE2000Loss(nn.Module):
    """Mean ΔE2000 over a (probe_mask & highlight_mask)-restricted region.

    Scales the result by `1 / 100` so the typical magnitude (~5) lands near
    the other losses and the configured weight stays in the (0.1, 1.0) band
    of the rest of the loss table.
    """

    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = float(scale)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Run the full ΔE2000 chain in fp32 — the formula has too many
        # sqrt / atan2 chains to be reliably fp16-safe by epsilons alone.
        # The output scalar autocasts back to fp16 in the AMP region.
        dE = delta_e2000_per_pixel_differentiable(pred.float(), target.float())
        return self.scale * _masked_mean(dE, mask)


