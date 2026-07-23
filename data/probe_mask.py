"""Build per-scene probe-region masks from meta.json.

In MIT-MI, every scene includes a chrome sphere and a gray sphere planted in
view as illumination references. Pixels inside those spheres are NOT scene
content — they are by-construction informative about the light, and a model
can shortcut illumination correction by getting probes right. For honest
metric reporting on val and test we mask them out.

The meta.json schema (top level):

    {
      "chrome": {"bounding_box": {"x", "y", "w", "h"},
                  "boundary_points": [{"x", "y"}, ...]},
      "gray":   {... same shape ...},
      ...
    }

Coordinates are in the original mip-0 frame. We convert to the target frame
(usually the model's training resolution) using the actual mip-2 image
dimensions as the intermediate reference (mip-2 = mip-0 / 4 exactly).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


_MIP2_DIVISOR = 4.0  # mip-2 is the level-2 mipmap: ÷4 in each axis.


def _polygon_to_xy(points: Sequence[dict]) -> np.ndarray | None:
    if not points:
        return None
    try:
        return np.asarray([[float(p["x"]), float(p["y"])] for p in points], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None


def _bbox_xywh(bbox: dict) -> tuple[float, float, float, float] | None:
    if not isinstance(bbox, dict):
        return None
    try:
        return float(bbox["x"]), float(bbox["y"]), float(bbox["w"]), float(bbox["h"])
    except (KeyError, TypeError, ValueError):
        return None


@lru_cache(maxsize=2048)
def _build_cached(
    meta_path: str,
    mip2_h: int,
    mip2_w: int,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    """Internal cached builder. Returns uint8 mask [target_h, target_w] (1=valid, 0=probe)."""
    import json

    mask = np.ones((target_h, target_w), dtype=np.uint8)
    p = Path(meta_path)
    if not p.exists():
        return mask

    try:
        meta = json.loads(p.read_text())
    except (OSError, ValueError):
        return mask

    # mip2 -> target scaling
    sx = target_w / float(mip2_w)
    sy = target_h / float(mip2_h)

    # Collect probe regions in the mip-2 frame, then dilate once at the end.
    probe_region = np.zeros((mip2_h, mip2_w), dtype=np.uint8)  # 1 = probe

    for key in ("chrome", "gray"):
        probe = meta.get(key)
        if not isinstance(probe, dict):
            continue

        poly = _polygon_to_xy(probe.get("boundary_points", []) or [])
        if poly is not None and len(poly) >= 3:
            pts_mip2 = (poly / _MIP2_DIVISOR).astype(np.int32)
            cv2.fillPoly(probe_region, [pts_mip2], color=1)
            continue

        bbox = _bbox_xywh(probe.get("bounding_box", {}) or {})
        if bbox is not None:
            bx, by, bw, bh = bbox
            x0 = int(np.floor(bx / _MIP2_DIVISOR))
            y0 = int(np.floor(by / _MIP2_DIVISOR))
            x1 = int(np.ceil((bx + bw) / _MIP2_DIVISOR))
            y1 = int(np.ceil((by + bh) / _MIP2_DIVISOR))
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(mip2_w, x1); y1 = min(mip2_h, y1)
            if x1 > x0 and y1 > y0:
                probe_region[y0:y1, x0:x1] = 1

    # Single dilation pass for anti-aliased rim margin (~0.5% of shorter side).
    margin = max(2, int(0.005 * min(mip2_h, mip2_w)))
    kernel = np.ones((margin, margin), dtype=np.uint8)
    probe_region = cv2.dilate(probe_region, kernel, iterations=1)
    mip2_mask = 1 - probe_region

    if (mip2_h, mip2_w) != (target_h, target_w):
        mask = cv2.resize(mip2_mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        mask = mip2_mask
    return mask


def build_probe_mask(
    meta_path: Path | str,
    mip2_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Build a binary mask covering the chrome+gray probe regions.

    Args:
        meta_path: Path to the scene's meta.json.
        mip2_shape: (H, W) of the mip-2 input image (the file we resize from).
        target_shape: (H, W) the dataset resizes to.

    Returns:
        uint8 array of shape (target_shape[0], target_shape[1]) with 1 for
        scene content and 0 for probe pixels. If meta.json is missing or has
        no probe info, returns all-ones (no masking).
    """
    return _build_cached(
        str(Path(meta_path)),
        int(mip2_shape[0]),
        int(mip2_shape[1]),
        int(target_shape[0]),
        int(target_shape[1]),
    ).copy()
