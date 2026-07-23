"""Edge-aware total variation on the illumination map L.

Penalizes |grad L| weighted by exp(-|grad I_in|/kappa) so that illumination
is smoothed in flat regions of the input and allowed to vary at edges. This
biases the R·L decomposition so that edges live in R while smooth fall-off
lives in L.
"""

from __future__ import annotations

import torch


def edge_aware_tv(
    L: torch.Tensor,
    input_rgb: torch.Tensor,
    kappa: float = 0.1,
) -> torch.Tensor:
    dy_in = (input_rgb[..., 1:, :] - input_rgb[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    dx_in = (input_rgb[..., :, 1:] - input_rgb[..., :, :-1]).abs().mean(dim=1, keepdim=True)
    wy = torch.exp(-dy_in / kappa)
    wx = torch.exp(-dx_in / kappa)

    dy_L = (L[..., 1:, :] - L[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    dx_L = (L[..., :, 1:] - L[..., :, :-1]).abs().mean(dim=1, keepdim=True)

    return (dy_L * wy).mean() + (dx_L * wx).mean()
