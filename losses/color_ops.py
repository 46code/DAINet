"""Color-space utility tensors used across losses.

Convention: all `pred` / `target` tensors entering losses are sRGB in [0, 1].
Loss helpers that need linear-RGB physics (chromaticity, log) convert through
`srgb_to_linear` first. This makes chromaticity ratios material-faithful (no
gamma distortion) and log differences physically meaningful.
"""

from __future__ import annotations

import torch


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """Standard sRGB EOTF (inverse gamma). Differentiable."""
    return torch.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)


def chromaticity(x: torch.Tensor, eps: float = 1e-3, linear: bool = True) -> torch.Tensor:
    """Per-pixel chromaticity (R/(R+G+B), G/(R+G+B), B/(R+G+B)).

    `linear=True` converts sRGB → linear first (the physically correct space
    for chromaticity ratios). Set `linear=False` for the sRGB-space variant.

    `eps` defaults to `1e-3` — above fp16's ~6e-5 subnormal floor so the
    `s.clamp_min(eps)` guard actually holds under AMP.
    """
    if linear:
        x = srgb_to_linear(x.clamp(0.0, 1.0))
    s = x.sum(dim=1, keepdim=True).clamp_min(eps)
    return x / s


# Backwards-compatible alias kept around for tests / external callers.
def srgb_to_chromaticity(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    return chromaticity(x, eps=eps, linear=True)


def luma_bt709(x: torch.Tensor) -> torch.Tensor:
    """Y channel via BT.709 weights (applied in whatever space x is in)."""
    return 0.2126 * x[:, 0:1] + 0.7152 * x[:, 1:2] + 0.0722 * x[:, 2:3]
