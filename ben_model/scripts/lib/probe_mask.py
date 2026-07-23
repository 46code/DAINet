"""Probe-mask helpers for the benchmark.

MIT-MI scenes contain chrome + gray probe spheres that must be excluded from
metric computation so no model is rewarded for "solving" the probe region.
The mit_mi test inputs here are *raw* (un-masked), so we build the mask from
the scene's ``meta.json`` using DAINet's own builder (identical to how
dainet masks during its evaluation). A black-hole fallback handles any scene
whose meta.json is missing.

The other three eval datasets (ambient6k, cl3an, wsrd24) have no probes, so
their mask is all-ones (full image scored).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config  # noqa: F401  (ensures DAINet is on sys.path)
from data.probe_mask import build_probe_mask as _bt_build_probe_mask


def mitmi_probe_mask(meta_path: Path, hw: tuple[int, int]) -> np.ndarray:
    """Probe mask (1=scene, 0=probe) for a mit_mi scene at resolution ``hw``.

    Args:
        meta_path: path to the scene's meta.json (mip-2 frame coords).
        hw: (H, W) of the image being scored.
    """
    h, w = int(hw[0]), int(hw[1])
    if not Path(meta_path).exists():
        return np.ones((h, w), dtype=np.float32)
    mask = _bt_build_probe_mask(meta_path, mip2_shape=(h, w), target_shape=(h, w))
    return mask.astype(np.float32)


def blackhole_mask(rgb01: np.ndarray, dark_thresh: float = 0.04) -> np.ndarray:
    """Fallback: detect large near-black discs (pre-masked probe holes)."""
    luma = rgb01.max(axis=-1)
    dark = (luma < dark_thresh).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel, iterations=2)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    keep = np.zeros_like(dark)
    for lbl in range(1, num):
        if stats[lbl, cv2.CC_STAT_AREA] >= 300:
            keep[labels == lbl] = 1
    return (1.0 - keep.astype(np.float32))


def full_mask(hw: tuple[int, int]) -> np.ndarray:
    return np.ones((int(hw[0]), int(hw[1])), dtype=np.float32)
