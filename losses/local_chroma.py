"""Edge-aware local-chroma cast removal on the corrected image I_out.

The 2026-05-28 failure mode: under a strong directional sidelight (e.g. a
red lamp from one side) the model fixes the *global* gray-world but leaves
a smooth *local* chromatic residue on the lit region — the wall's intensity
is restored but the lamp's colour is not removed from that segment.

The 2026-05-29 rework retired the old `local_chroma` (per-SAM-fusion-segment
chroma variance on R) because SAM-fusion ids are object-level, not
chroma-level, so two materials sharing one id were forced to the same
chroma — a direct fight with reconstruction. This re-introduces a
*non-fighting* local-chroma lever instead:

    Edge-aware chromaticity total variation on **I_out**, gated by the
    *input* luminance edges. Chroma gradients are penalised only where the
    input has no luminance edge (`exp(-|∇luma_in|/κ)` weight), so a smooth
    colour cast is suppressed while genuine material / chroma boundaries —
    which coincide with input luminance edges — are preserved.

Because it operates on the predicted image's chromaticity over a guide that
the prediction is free to honour (it never forces two GT-distinct chromas
together), it cannot fight reconstruction the way the retired R+SAM variant
did. Off in the lean baseline (weight 0); enabled in the `robust` experiment
and as ablation row A-localchroma. See docs/dainet_losses.md.

fp32-internal (chromaticity divide + exp) so it is bf16-AMP safe.
"""

from __future__ import annotations

import torch

from .color_ops import chromaticity, luma_bt709


def local_chroma_tv(
    I_out: torch.Tensor,
    input_rgb: torch.Tensor,
    kappa: float = 0.1,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Edge-aware chroma-TV on I_out, guided by input luminance edges.

    Args:
        I_out: predicted corrected image, sRGB [B, 3, H, W] in [0, 1].
        input_rgb: the lit input, sRGB [B, 3, H, W] in [0, 1] (edge guide).
        kappa: edge-weight temperature. Smaller → more boundaries preserved.
        eps: chromaticity denominator floor (>= linear-space subnormal floor).

    Returns:
        Scalar mean edge-weighted chroma total variation.
    """
    chroma = chromaticity(I_out, eps=eps, linear=True).float()  # [B, 3, H, W]
    luma = luma_bt709(input_rgb).float()  # [B, 1, H, W]

    # Input luminance gradients → edge-preserving weights.
    dy_in = (luma[..., 1:, :] - luma[..., :-1, :]).abs()
    dx_in = (luma[..., :, 1:] - luma[..., :, :-1]).abs()
    wy = torch.exp(-dy_in / kappa)
    wx = torch.exp(-dx_in / kappa)

    # Chromaticity gradients on the prediction (mean across the 3 channels).
    dy_c = (chroma[..., 1:, :] - chroma[..., :-1, :]).abs().mean(dim=1, keepdim=True)
    dx_c = (chroma[..., :, 1:] - chroma[..., :, :-1]).abs().mean(dim=1, keepdim=True)

    return (dy_c * wy).mean() + (dx_c * wx).mean()
