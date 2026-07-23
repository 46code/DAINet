"""Global illuminant angular supervision (training-only aux).

Supervises the optional ``IlluminantHead`` (``out["illuminant"]``) toward
the scene's true illuminant chromaticity — the colour cast the model must
remove. The GT illuminant direction is the per-image ratio of the input's
mean linear RGB to the target's mean linear RGB (the gray-world estimate of
the lighting colour): ``illum_gt = mean_lin(input) / mean_lin(target)``.

The loss is ``1 − cos(pred, illum_gt)`` between the two unit-normalised
chromaticity directions — smooth, bounded in [0, 2], scale-invariant, and
minimised iff the predicted illuminant direction matches the GT direction.
It is proportional to the (squared-half-)angle for small angles — a direct,
training-only color-constancy supervision — without the ``acos`` gradient
blow-up near ±1.

Off in the lean baseline; enabled in the `robust` experiment + the B-arch
ablation rows. fp32-internal (linear-RGB + normalise) → bf16-AMP safe.
"""

from __future__ import annotations

import torch

from .color_ops import srgb_to_linear


def _masked_mean_rgb(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Per-image mean RGB over valid pixels → [B, 3]."""
    if mask is None:
        return x.mean(dim=[2, 3])
    m = mask
    if m.dim() == 3:
        m = m.unsqueeze(0)
    m = m.to(dtype=x.dtype, device=x.device)
    m = m.expand(x.shape[0], 1, x.shape[2], x.shape[3])
    denom = m.sum(dim=[2, 3]).clamp_min(1.0)  # [B, 1]
    return (x * m).sum(dim=[2, 3]) / denom  # [B, 3]


def illuminant_angular_loss(
    illum_pred: torch.Tensor,
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
    eps: float = 1e-4,
) -> torch.Tensor:
    """1 − cosine between predicted and GT illuminant chromaticity directions.

    Args:
        illum_pred: [B, 3] positive illuminant chromaticity from IlluminantHead.
        input_rgb:  [B, 3, H, W] sRGB lit input.
        target:     [B, 3, H, W] sRGB flat-lit target.
        mask:       optional [B,1,H,W] probe/highlight mask.
    """
    p_lin = srgb_to_linear(input_rgb.clamp(0.0, 1.0)).float()
    t_lin = srgb_to_linear(target.clamp(0.0, 1.0)).float()
    mp = _masked_mean_rgb(p_lin, mask)  # [B, 3]
    mt = _masked_mean_rgb(t_lin, mask).clamp_min(eps)
    illum_gt = mp / mt
    illum_gt = illum_gt / illum_gt.norm(dim=1, keepdim=True).clamp_min(eps)

    pred = illum_pred.float()
    pred = pred / pred.norm(dim=1, keepdim=True).clamp_min(eps)

    cos = (pred * illum_gt).sum(dim=1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return (1.0 - cos).mean()
