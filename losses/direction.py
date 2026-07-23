"""Light-direction regression loss for the DirectionHead (contribution C).

Supervises the model's predicted ``(φ, θ, b)`` encoding against the GT capture
metadata so the head learns to recover the lighting direction from the image
alone. Training-only: gated by ``has_meta`` (a real photo has no GT direction).

The prediction and target share the ``IlluminationEmbedding.encode_raw`` layout
``[sin φ, cos φ, sin θ, cos θ, log b]``. The two ``(sin, cos)`` pairs are unit
vectors on the circle, so the angular error is ``1 − cos(Δ)`` per pair (their
dot product); ``log b`` is matched with an L1 term.
"""

from __future__ import annotations

import torch

from models.illum_embedding import IlluminationEmbedding


def direction_pred_loss(
    pred_enc: torch.Tensor,
    phi: torch.Tensor,
    theta: torch.Tensor,
    bnorm: torch.Tensor,
    has_meta: torch.Tensor | None = None,
) -> torch.Tensor:
    """pred_enc: [B, 5] from DirectionHead; (phi, theta, bnorm): [B] GT. Returns a scalar."""
    gt = IlluminationEmbedding.encode_raw(phi, theta, bnorm).to(pred_enc.dtype)  # [B, 5]
    phi_cos = (pred_enc[:, 0:2] * gt[:, 0:2]).sum(dim=-1)    # [B] in [-1, 1]
    theta_cos = (pred_enc[:, 2:4] * gt[:, 2:4]).sum(dim=-1)  # [B]
    angular = (1.0 - phi_cos) + (1.0 - theta_cos)            # [B] in [0, 4]
    log_b = (pred_enc[:, 4] - gt[:, 4]).abs()                # [B]
    per_sample = angular + log_b                             # [B]
    if has_meta is not None:
        m = has_meta.to(per_sample.dtype)
        return (per_sample * m).sum() / m.sum().clamp_min(1.0)
    return per_sample.mean()
