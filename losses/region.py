"""Material-segment-aware regularization on chromaticity.

`region_chroma_variance`: within each segment of the predicted image, the
chromaticity (R/(R+G+B), G/(...), B/(...)) should have low variance — i.e. a
wall painted one color should map to one color after correction. Vectorized
via a `scatter_add` implementation.
"""

from __future__ import annotations

import torch

from .color_ops import chromaticity


def per_segment_stats(
    values: torch.Tensor, segments: torch.Tensor, n_max: int = 256
) -> dict[str, torch.Tensor]:
    """Vectorized per-segment mean and second-moment.

    Args:
        values: [B, C, H, W]
        segments: [B, 1, H, W] int
        n_max: maximum number of distinct segment ids tracked per image.

    Returns:
        dict with:
            means:    [B, C, n_max]
            sq_means: [B, C, n_max]
            counts:   [B, n_max]
    """
    B, C, _, _ = values.shape
    seg_flat = segments.long().clamp(0, n_max - 1).view(B, -1)
    vals_flat = values.reshape(B, C, -1)

    counts = torch.zeros(B, n_max, device=values.device, dtype=values.dtype)
    counts.scatter_add_(
        1, seg_flat, torch.ones_like(seg_flat, dtype=values.dtype)
    )

    seg_expanded = seg_flat.unsqueeze(1).expand(-1, C, -1)
    sums = torch.zeros(B, C, n_max, device=values.device, dtype=values.dtype)
    sums.scatter_add_(2, seg_expanded, vals_flat)
    sq_sums = torch.zeros(B, C, n_max, device=values.device, dtype=values.dtype)
    sq_sums.scatter_add_(2, seg_expanded, vals_flat * vals_flat)

    counts_safe = counts.clamp_min(1.0).unsqueeze(1)
    means = sums / counts_safe
    sq_means = sq_sums / counts_safe
    return {"means": means, "sq_means": sq_means, "counts": counts}


_LOG_CHROMA_EPS = 1e-3


def region_chroma_variance(
    pred: torch.Tensor,
    segments: torch.Tensor,
    n_max: int = 256,
    eps: float = 1e-2,
) -> torch.Tensor:
    # Variance is measured in *log-chromaticity* space rather than raw
    # chromaticity. Raw chromaticity lives on the 2-simplex and saturates
    # near zero variance very quickly (the per-segment loss flatlines),
    # which collapses the gradient. log-chrom space does not saturate the
    # same way, so within-segment differences keep producing signal even
    # when the linear-chromaticity spread is already small.
    chroma = chromaticity(pred, eps=eps, linear=True)
    log_chroma = torch.log(chroma.clamp_min(_LOG_CHROMA_EPS))
    stats = per_segment_stats(log_chroma, segments, n_max=n_max)
    var_per_seg = (stats["sq_means"] - stats["means"] ** 2).clamp_min(0.0)
    # Sum across the 3 chroma channels, weight by segment size.
    weighted = var_per_seg.sum(dim=1) * stats["counts"]  # [B, n_max]
    total_weight = stats["counts"].sum().clamp_min(1.0)
    return weighted.sum() / total_weight


def region_active_segment_count(
    segments: torch.Tensor, n_max: int = 256, min_size: int = 1
) -> int:
    """Number of distinct segments in the batch with at least ``min_size``
    pixels. Diagnostic used by the loss manager to surface
    ``train/region/region_segments_active`` to wandb — confirms the
    region_var gradient signal is alive."""
    B = segments.shape[0]
    seg_flat = segments.long().clamp(0, n_max - 1).view(B, -1)
    counts = torch.zeros(B, n_max, device=segments.device, dtype=torch.float32)
    counts.scatter_add_(1, seg_flat, torch.ones_like(seg_flat, dtype=torch.float32))
    return int((counts >= float(min_size)).sum().item())
