"""L2 between model-predicted SH coefficients and chrome-probe SH targets.

Samples without a chrome probe (`has_sh=False`) contribute zero to the loss.
"""

from __future__ import annotations

import torch


def probe_sh_loss(
    sh_pred: torch.Tensor, sh_target: torch.Tensor, has_sh: torch.Tensor
) -> torch.Tensor:
    if has_sh.sum() == 0:
        return torch.zeros((), device=sh_pred.device, dtype=sh_pred.dtype)
    mask = has_sh.to(sh_pred.dtype).view(-1, 1, 1)
    sq = (sh_pred - sh_target) ** 2 * mask
    denom = mask.sum() * sh_pred.shape[1] * sh_pred.shape[2]
    return sq.sum() / denom.clamp_min(1.0)
