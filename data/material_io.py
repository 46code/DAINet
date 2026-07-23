"""MIT-MI material-mask loader and taxonomy.

The MIT-Multi-Illumination dataset ships a hand-labeled `materials_mip2.png`
per scene (uint8, sparse class ids drawn from a 36-class taxonomy; ~6–11
distinct ids per scene). This module:

1. Loads the dataset-local taxonomy table at
   `data/raw/mit_mi/material_taxonomy.json` and builds a 256-entry LUT
   mapping raw ids → contiguous training class ids in `[0, K_material)`.
   Unknown raw ids map to `IGNORE_INDEX = 255`.
2. Loads a material PNG, applies the LUT, and returns `[H, W] int32`.

Used by:
- `data.dataset.DAINetDataset` (training/val only — never test/inference).
- `losses.material.material_ce_loss` (consumes `int32 [B, 1, H, W]`).
- `scripts.build_material_taxonomy` writes the JSON.
- `scripts.build_sam_fusion_centroids` reads the JSON for the
  material-anchored K-means initialization.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


IGNORE_INDEX = 255


@lru_cache(maxsize=4)
def load_taxonomy(path: str | Path) -> dict[str, object]:
    """Return the parsed taxonomy JSON.

    Schema:
        {
          "raw_to_contiguous": {"3": 0, "7": 1, ...},
          "K": <int — number of contiguous training classes>
        }
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Material taxonomy missing at {p}. Build with "
            "`python scripts/build_material_taxonomy.py --root data/raw/mit_mi/jpg`."
        )
    with open(p) as f:
        tx = json.load(f)
    # Normalize keys to ints in the in-memory copy.
    raw_to_contig = {int(k): int(v) for k, v in tx["raw_to_contiguous"].items()}
    return {"raw_to_contiguous": raw_to_contig, "K": int(tx["K"])}


@lru_cache(maxsize=4)
def _lut_for(path: str) -> np.ndarray:
    tx = load_taxonomy(path)
    lut = np.full(256, IGNORE_INDEX, dtype=np.int32)
    for raw, contig in tx["raw_to_contiguous"].items():
        lut[int(raw)] = int(contig)
    return lut


def num_classes(taxonomy_path: str | Path) -> int:
    return int(load_taxonomy(taxonomy_path)["K"])


@lru_cache(maxsize=32)
def _load_remapped(png_path: str, taxonomy_path: str) -> np.ndarray | None:
    p = Path(png_path)
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    lut = _lut_for(taxonomy_path)
    return lut[img]  # int32 [H, W]


def load_material_mask(
    png_path: str | Path,
    taxonomy_path: str | Path,
) -> np.ndarray | None:
    """Return a remapped material mask as `[H, W] int32`, or None if missing.

    Pixels whose raw id is not in the taxonomy map to `IGNORE_INDEX` (255).
    """
    out = _load_remapped(str(Path(png_path)), str(Path(taxonomy_path)))
    if out is None:
        return None
    return out.copy()  # the cached array is shared; defensive copy


def material_dominant_id(
    material_mask: np.ndarray,
    raw_ids: np.ndarray,
    n_raw: int,
) -> np.ndarray:
    """Per equivalence-class segment, return the dominant material class.

    Args:
        material_mask: [H, W] int32 remapped material ids; IGNORE_INDEX
            for unknown pixels.
        raw_ids:       [H, W] int32 equivalence-class segment ids.
        n_raw:         raw_ids.max() + 1.

    Returns:
        dominant: [n_raw] int32. Each entry is the majority-vote material
            class for that segment, or IGNORE_INDEX if the segment is
            entirely covered by ignored pixels.
    """
    if material_mask.shape != raw_ids.shape:
        raise ValueError(
            f"shape mismatch: material {material_mask.shape} vs raw {raw_ids.shape}"
        )
    # Build a [n_raw, K+1] histogram where the last column is the ignore bucket.
    mat = material_mask.reshape(-1).astype(np.int64)
    seg = raw_ids.reshape(-1).astype(np.int64)
    # Determine K from the max non-ignore value present (callers can pass an
    # explicit K via num_classes() but we infer here for robustness).
    valid = mat != IGNORE_INDEX
    K = int(mat[valid].max()) + 1 if valid.any() else 0
    if K == 0:
        return np.full(n_raw, IGNORE_INDEX, dtype=np.int32)

    # Compact mat: ignore-pixels go to column K.
    col = np.where(valid, mat, K).astype(np.int64)
    flat_key = seg * (K + 1) + col
    hist = np.bincount(flat_key, minlength=n_raw * (K + 1)).reshape(n_raw, K + 1)
    # Argmax over real classes only; if a segment's real-class mass is 0 it
    # falls back to IGNORE_INDEX.
    real_hist = hist[:, :K]
    has_real = real_hist.sum(axis=1) > 0
    dominant = np.where(has_real, real_hist.argmax(axis=1), IGNORE_INDEX)
    return dominant.astype(np.int32)


def _clear_caches_for_test() -> None:
    """Test hook: drop the lru_caches so monkey-patched taxonomies apply."""
    load_taxonomy.cache_clear()
    _lut_for.cache_clear()
    _load_remapped.cache_clear()
