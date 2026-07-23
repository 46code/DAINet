"""Cross-direction relighting consistency loss — contribution B (novelty).

For every same-scene (i, j) pair in the batch, enforce::

    R[j] · L[i]  ≈  input(i)

and its symmetric counterpart. Compared to ``dir_consistency_R`` (which
only asks R[i] ≈ R[j]) this loss directly tests the R · L
factorization across directions: it forces L[i] to capture the
direction-specific lighting field and R[j] to be a true reflectance,
because swapping L between same-scene directions must reconstruct the
corresponding input image.

``ScenePairBatchSampler`` guarantees ≥ 1 same-scene pair per batch, so
this loss almost always contributes (returns 0 only on the
near-impossible no-pair case).

Smooth softclamp (2026-05-28 rework)
------------------------------------
Earlier versions used a straight-through hard clamp at 1.0 to keep the
forward output in [0, 1]. That STE saturated once R · L hit 1.0 across
most pixels and the loss plateaued at ~1.5e-2 from step 1575 onward —
gradient flowed but the *value* stopped moving so the term contributed
no meaningful signal.

The new variant is a smooth exponential softclamp::

    softclamp(x; τ) = τ · (1 − exp(−x / τ))

It is exactly the identity for small x, asymptotes to τ for large x,
and is smooth everywhere with a non-vanishing derivative. With τ = 1.2
the cost surface stays informative past the [0, 1] ceiling so the
factorisation keeps improving when R · L is briefly > 1.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Softclamp ceiling. Slightly above 1 so the loss has headroom past the
# nominal output range — keeps gradient alive when R · L overshoots
# during early training.
_SOFTCLAMP_TAU: float = 1.2


def _softclamp(x: torch.Tensor, tau: float = _SOFTCLAMP_TAU) -> torch.Tensor:
    """Smooth top-clamp: x ≈ x for small x, asymptotes to tau for large x.

    `1 − exp(−x / τ)` is C∞, identity-tangent at the origin, and has a
    strictly positive derivative everywhere — no STE-style saturation.
    """
    return tau * (1.0 - torch.exp(-x.clamp_min(0.0) / tau))


def xdir_relighting_loss(
    R: torch.Tensor,         # [B, 3, H, W] predicted reflectance
    L: torch.Tensor,         # [B, 3, H, W] predicted illumination
    input_rgb: torch.Tensor, # [B, 3, H, W] the *input* image (sRGB)
    scenes: list[str],
) -> torch.Tensor:
    if len(scenes) < 2:
        return torch.zeros((), device=R.device, dtype=torch.float32)
    losses: list[torch.Tensor] = []
    R_f = R.float()
    L_f = L.float()
    in_f = input_rgb.float()
    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            if scenes[i] != scenes[j]:
                continue
            synth_ij = _softclamp(R_f[j] * L_f[i])
            synth_ji = _softclamp(R_f[i] * L_f[j])
            losses.append(F.l1_loss(synth_ij, in_f[i]))
            losses.append(F.l1_loss(synth_ji, in_f[j]))
    if not losses:
        return torch.zeros((), device=R.device, dtype=torch.float32)
    return torch.stack(losses).mean()
