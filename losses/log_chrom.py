"""Log-chromaticity loss: L1 in log linear-RGB.

Operating in linear RGB (after `srgb_to_linear`) makes the log-differences
correspond to multiplicative-illumination ratios, which is what the network's
illumination head L is implicitly learning.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .color_ops import srgb_to_linear


def log_chromaticity_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-4
) -> torch.Tensor:
    p_lin = srgb_to_linear(pred.clamp(0.0, 1.0))
    t_lin = srgb_to_linear(target.clamp(0.0, 1.0))
    # log of small linear-RGB values has high local gradient even in bf16
    # (1/eps ≈ 1e4 at the floor). Force fp32 here so the chain is bit-stable
    # regardless of the outer autocast dtype.
    return F.l1_loss(
        torch.log(p_lin.float().clamp_min(eps)),
        torch.log(t_lin.float().clamp_min(eps)),
    )
