"""SAM2 segmentation: cache reader + on-the-fly generator.

Two entry points:

- ``load_sam_mask(path)`` — read a precomputed mask written by
  ``scripts/precompute_sam.py``. Used by the dataset during training
  and validation (cache hit is mandatory; see ``data/dataset.py``).
- ``segment_image(rgb_uint8)`` — run SAM2 inline on a single image,
  then *project* the AMG segments into the same K_fusion id space used
  during training. Used by ``scripts/infer.py`` for arbitrary online
  data where no cache file exists.

The K_fusion projection (frozen centroids saved to
``data/raw/mit_mi/sam_fusion_centroids.npy`` by
``scripts/build_sam_fusion_centroids.py``) is the key reason inference
and training share the SAME segmentation distribution: the global K-means
output is deterministic and viewpoint-symmetric, so the
SegmentationEncoder never sees a different id-space at deploy time than
it did at training time. If the centroids file is missing we fall back
to raw equivalence-class ids and log a warning — useful for unit tests
and dev environments where the centroids haven't been built yet.

SAM3 was evaluated but is text-prompted and ships no AutomaticMaskGenerator,
so it cannot produce the exhaustive per-scene id maps this pipeline
expects — SAM2's AMG is the right fit.
"""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


_SAM2_MODEL = None
_SAM2_DEVICE = None
_DEFAULT_WEIGHTS = "~/my_model/sam2_weights/sam2.1_hiera_large.pt"
_DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
_DEFAULT_CENTROIDS = "data/raw/mit_mi/sam_fusion_centroids.npy"


@lru_cache(maxsize=32)
def _load_cached(path_str: str) -> np.ndarray | None:
    """Read an id map from disk.

    Accepts ``.npy`` (preferred, written by the current precompute scripts)
    or ``.png`` (legacy caches written by older runs). Either suffix may be
    passed in — the loader resolves to whichever sibling exists. Returns
    ``int32 [H, W]`` or ``None`` if no file exists at either path.
    """
    p = Path(path_str)
    npy_path = p if p.suffix == ".npy" else p.with_suffix(".npy")
    if npy_path.exists():
        arr = np.load(str(npy_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        return arr.astype(np.int32)
    png_path = p if p.suffix == ".png" else p.with_suffix(".png")
    if not png_path.exists():
        return None
    img = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.int32)


def load_sam_mask(path: Path | str) -> np.ndarray | None:
    """Return a [H, W] int32 SAM2 segment-id array, or None if missing.

    Caller may pass either ``…/sam_mip2.npy`` (current contract) or
    ``…/sam_mip2.png`` (legacy); the loader transparently resolves to
    whichever exists on disk.
    """
    return _load_cached(str(Path(path)))


def load_chroma_superpixels(path: Path | str) -> np.ndarray | None:
    """Return a [H, W] int32 chromaticity super-pixel id array, or None.

    Same ``.npy`` / ``.png`` fallback rule as `load_sam_mask`.
    """
    return _load_cached(str(Path(path)))


@lru_cache(maxsize=4)
def _load_centroids(path_str: str) -> np.ndarray | None:
    p = Path(path_str).expanduser()
    if not p.exists():
        return None
    return np.load(str(p)).astype(np.float32)


def get_sam2_segmenter(
    weights_path: Path | str | None = None,
    config: str | None = None,
    *,
    device: str | None = None,
):
    """Lazy-load the SAM2 automatic mask generator. Singleton.

    ``device`` (e.g. ``"cuda:1"``) is forwarded to ``build_sam2``; when None,
    SAM2's own default (``"cuda"`` → cuda:0) is used. The singleton is rebuilt
    if a different device is requested so the segmenter follows the job's GPU
    rather than silently pinning to cuda:0.
    """
    global _SAM2_MODEL, _SAM2_DEVICE
    if _SAM2_MODEL is not None and (device is None or device == _SAM2_DEVICE):
        return _SAM2_MODEL
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "SAM2 not installed. Run: "
            "git clone https://github.com/facebookresearch/sam2 ~/my_model/sam2 && "
            "pip install -e ~/my_model/sam2"
        ) from exc
    p = Path(weights_path or _DEFAULT_WEIGHTS).expanduser()
    if not p.exists():
        raise FileNotFoundError(
            f"SAM2 weights not found at {p}. Download a SAM2 checkpoint "
            "from https://github.com/facebookresearch/sam2#model-description."
        )
    build_kwargs = {"apply_postprocessing": False}
    if device is not None:
        build_kwargs["device"] = device
    sam = build_sam2(config or _DEFAULT_CONFIG, str(p), **build_kwargs)
    _SAM2_MODEL = SAM2AutomaticMaskGenerator(sam)
    _SAM2_DEVICE = device
    return _SAM2_MODEL


def segment_image(
    rgb_uint8: np.ndarray,
    weights_path: Path | str | None = None,
    *,
    centroids_path: Path | str | None = None,
    min_area: int = 64,
    id_offset: int = 1,
    device: str | None = None,
) -> np.ndarray:
    """Run SAM2 on a single H×W×3 uint8 RGB image and project to K_fusion ids.

    The returned id map lives in the same K_fusion id space used at
    training time (when centroids are available on disk). Without
    centroids, falls back to raw equivalence-class ids and warns.

    Returns: [H, W] int32 with values in [0, K_fusion] (or [0, N_raw] in
    the fallback case).
    """
    # Imported here to keep this module importable without numpy/cv2-heavy
    # fusion deps at startup if `segment_image` is never called.
    from .sam_fusion import (
        fuse_equivalence_classes,
        compute_segment_features,
        assign_to_centroids,
        per_view_id_map,
        rasterize_segment_labels,
    )

    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    seg = get_sam2_segmenter(weights_path=weights_path, device=device)
    masks = seg.generate(rgb_uint8)
    H, W = rgb_uint8.shape[:2]
    per_view = per_view_id_map(masks, (H, W))[None, ...]  # [1, H, W]
    raw_ids = fuse_equivalence_classes(per_view, min_area=min_area)

    centroids = _load_centroids(str(centroids_path or _DEFAULT_CENTROIDS))
    if centroids is None:
        warnings.warn(
            f"SAM fusion centroids not found at "
            f"{centroids_path or _DEFAULT_CENTROIDS}; falling back to raw "
            "equivalence-class ids. Train/inference id-spaces will differ. "
            "Run scripts/build_sam_fusion_centroids.py.",
            RuntimeWarning,
        )
        return raw_ids.astype(np.int32)

    features = compute_segment_features(raw_ids, rgb_uint8)
    labels = assign_to_centroids(features, centroids)
    sam_map = rasterize_segment_labels(raw_ids, labels, id_offset=id_offset)
    return sam_map.astype(np.int32)


def _reset_sam2_for_test() -> None:
    """Test hook: clear the singleton so monkey-patched segmenters apply."""
    global _SAM2_MODEL, _SAM2_DEVICE
    _SAM2_MODEL = None
    _SAM2_DEVICE = None
    _load_centroids.cache_clear()
