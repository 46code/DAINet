"""Material auxiliary losses (training-only).

Two terms, both gated on ``batch['has_material']`` so the loss is exactly
zero on a batch where no scene supplied a usable materials_mip2.png:

1. ``material_ce_loss`` — pixel cross-entropy between the network's
   auxiliary material head logits and the remapped MIT-MI material ids
   (IGNORE_INDEX=255 pixels are skipped). Optionally masks probe pixels.

2. ``material_R_variance_loss`` — within-material reflectance R should
   be uniform. Reuses ``per_segment_stats`` from ``losses.region`` to
   compute per-segment mean / sq-mean of R and sums the variance, with
   the ignore-index pushed into a "trash" bucket that gets dropped.

Both losses operate on linear-RGB R when supplied; callers don't need to
pre-convert. If the trainer fires classifier-free dropout it should
zero ``has_material`` (same pattern as ``has_sh``) so the null code
path doesn't waste gradient on supervision the model can't see.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .region import per_segment_stats


IGNORE_INDEX = 255


def _broadcast_flag(flag: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast a [B] bool/scalar flag to [B, 1, 1, 1] matching ref's dtype."""
    if flag.dim() == 0:
        flag = flag.unsqueeze(0)
    return flag.to(ref.device, dtype=ref.dtype).view(-1, 1, 1, 1)


def material_ce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    probe_mask: torch.Tensor | None,
    has_material: torch.Tensor,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Pixel CE on material logits, gated by ``has_material``.

    Args:
        logits:        [B, K, H, W] float — material head output.
        target:        [B, 1, H, W] int — remapped material ids, with
                       ignore_index where the dataset can't supervise.
        probe_mask:    [B, 1, H, W] or None — 1 where pixel is scene
                       content, 0 over probes. Probe pixels are skipped.
        has_material:  [B] bool — false batches contribute zero.

    Returns:
        scalar loss. If every batch element has has_material=False or
        every pixel is ignored, returns zero (still attached to the
        compute graph so .backward is safe).
    """
    if logits.dim() != 4:
        raise ValueError(f"logits must be [B,K,H,W], got {logits.shape}")
    B, K, H, W = logits.shape
    tgt = target
    if tgt.dim() == 4 and tgt.shape[1] == 1:
        tgt = tgt[:, 0]
    tgt = tgt.long()
    # Per-pixel CE, no reduction. ignore_index zeros gradient for those pixels.
    ce = F.cross_entropy(logits, tgt, reduction="none", ignore_index=ignore_index)
    # Build a per-pixel mask: (target != ignore) ∧ probe_mask ∧ has_material.
    valid = (tgt != ignore_index).to(ce.dtype)
    if probe_mask is not None:
        pm = probe_mask
        if pm.dim() == 4 and pm.shape[1] == 1:
            pm = pm[:, 0]
        valid = valid * pm.to(ce.dtype)
    has_m = has_material
    if has_m.dim() == 0:
        has_m = has_m.unsqueeze(0)
    has_m = has_m.to(ce.dtype).view(-1, 1, 1)
    valid = valid * has_m

    denom = valid.sum().clamp_min(1.0)
    return (ce * valid).sum() / denom


def material_R_variance_loss(
    R: torch.Tensor,
    material_seg: torch.Tensor,
    has_material: torch.Tensor,
    num_classes: int,
    ignore_index: int = IGNORE_INDEX,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Within-material variance of the reflectance map R.

    Args:
        R:            [B, 3, H, W] reflectance prediction.
        material_seg: [B, 1, H, W] int ids in [0, K) ∪ {ignore_index}.
        has_material: [B] bool batch-level flag.
        num_classes:  K (number of real classes).
        ignore_index: id treated as "no supervision"; mapped to a trash bin.

    Returns:
        scalar loss. Per-segment variance summed across channels, count-
        weighted, normalized by total valid pixels. Zero when every batch
        element has has_material=False or every pixel is ignored.
    """
    if R.dim() != 4 or R.shape[1] != 3:
        raise ValueError(f"R must be [B,3,H,W], got {R.shape}")
    K = int(num_classes)
    if K <= 0:
        return torch.zeros((), device=R.device, dtype=R.dtype)

    # Push ignore-index into the trash slot at id=K, so the call to
    # per_segment_stats (which clamps to [0, n_max-1]) gathers ignored
    # pixels into one bucket we then drop below.
    seg = material_seg.clone().long()
    seg[seg == ignore_index] = K  # trash bin slot
    seg = seg.clamp_(0, K)

    stats = per_segment_stats(R, seg, n_max=K + 1)
    # variance = E[x^2] - E[x]^2, clamped non-negative for fp stability.
    var_per_seg = (stats["sq_means"] - stats["means"] ** 2).clamp_min(0.0)
    counts = stats["counts"]  # [B, K+1]
    # Drop the trash-bin column so ignore pixels contribute nothing.
    counts = counts[:, :K]
    var_per_seg = var_per_seg[:, :, :K]
    # Sum variance across the 3 channels, weight by segment size.
    weighted = var_per_seg.sum(dim=1) * counts  # [B, K]
    # Apply the per-batch has_material gate.
    has_m = has_material
    if has_m.dim() == 0:
        has_m = has_m.unsqueeze(0)
    has_m = has_m.to(R.dtype).view(-1, 1)
    weighted = weighted * has_m
    counts = counts * has_m
    total = counts.sum().clamp_min(1.0)
    return weighted.sum() / (total + eps)
