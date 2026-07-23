"""Metric backbone for the benchmark.

Thin wrapper over DAINet's own ``evaluation.metrics`` so every model
(including dainet) is scored with the *identical* implementation dainet uses in
its paper. No-reference NIQE/BRISQUE are optional extras.

All inputs are [B,3,H,W] float tensors in [0,1]; mask is [B,1,H,W] (1=valid).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from . import config  # noqa: F401  (puts DAINet on sys.path)
from evaluation.metrics import (
    psnr, ssim, ms_ssim, lpips_score,
)

# The ben_guide.md metric family (order used in tables).
PRIMARY_KEYS = [
    "psnr", "ms_ssim", "lpips",
]
# Higher-is-better metrics (everything else is lower-is-better).
HIGHER_BETTER = {"psnr", "ms_ssim", "ssim"}


@torch.no_grad()
def score_pair(pred: torch.Tensor, target: torch.Tensor,
               mask: Optional[torch.Tensor] = None,
               with_lpips: bool = True) -> dict:
    """Full per-image metric dict for one [1,3,H,W] prediction/target pair."""
    pred = pred.clamp(0, 1)
    target = target.clamp(0, 1)
    row = {
        "psnr": float(psnr(pred, target, mask=mask).mean().item()),
        "ssim": float(ssim(pred, target, mask=mask).mean().item()),
        "ms_ssim": float(ms_ssim(pred, target, mask=mask).mean().item()),
    }
    row["lpips"] = (float(lpips_score(pred, target, net="alex", mask=mask).mean().item())
                    if with_lpips else float("nan"))
    return row


def aggregate(per_sample: list[dict], keys: list[str]) -> dict:
    """mean/std/median/p95/count per metric over a list of per-sample dicts."""
    agg = {}
    for k in keys:
        vals = np.array([r[k] for r in per_sample
                         if k in r and not np.isnan(r[k])], dtype=np.float64)
        if vals.size == 0:
            continue
        agg[k] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
            "p95": float(np.percentile(vals, 95)),
            "count": int(vals.size),
        }
    return agg
