"""Two-stage SAM2 multi-view fusion + K-means semantic compression.

Stage A — per-scene multi-view oversegmentation:
    Run SAM2 AMG on all V (≈25) directional views of a scene; for each pixel
    form a V-tuple of (per-view mask ids); pixels with identical tuples form
    one raw fused segment. Output: int32 raw-id map + per-segment 6-D feature
    vectors written next to the cache.

Stage B (one-shot, see scripts/build_sam_fusion_centroids.py) — global
K-means over all training-scene segments produces a single fixed centroid
matrix `[K_fusion, 6]` saved to disk.

Stage C — per-scene rasterization: load raw ids + features + centroids,
assign each segment to its nearest centroid, write `sam_mip2.png` (uint16,
ids in `[0, K_fusion]`).

At inference, `data.segmentation.segment_image` runs the same Stage A + C
on a single image so the train and inference id spaces match exactly.

The on-disk contract (`sam_mip2.png` uint16, read back via
`load_sam_mask` as `int32 [H, W]`) is preserved. The SegmentationEncoder
contract (`[B, 1, H, W] int32 -> [B, embed_dim]`) is preserved.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


N_FEATURES = 6  # (Lab a, Lab b, cx, cy, log_area_frac, L_norm)

# Disk I/O perf knobs. The readback sanity check in `write_id_map_png` is
# good paranoia but doubles I/O on every write; the precompute opts in via
# its own CLI flag (`--verify-png` -> `set_verify_png_write(True)`).
PNG_COMPRESSION = 1  # libpng level 1 is ~3x faster encode than 3, ~10% larger
_VERIFY_PNG_WRITE = False


def set_verify_png_write(enabled: bool) -> None:
    """Toggle the post-write readback sanity check in `write_id_map_png`."""
    global _VERIFY_PNG_WRITE
    _VERIFY_PNG_WRITE = bool(enabled)


def per_view_id_map(masks: list[dict], shape: tuple[int, int]) -> np.ndarray:
    """Flatten a list of AMG mask dicts to a [H, W] uint16 id map.

    Larger masks first, so smaller masks overwrite (matches the AMG intent
    of nesting fine detail inside coarse regions). id 0 = background (no
    AMG mask covers that pixel).
    """
    H, W = shape
    out = np.zeros((H, W), dtype=np.uint16)
    masks_sorted = sorted(masks, key=lambda m: -int(m.get("area", 0)))
    for i, m in enumerate(masks_sorted, start=1):
        if i >= 65535:
            break
        seg = m["segmentation"].astype(bool)
        out[seg] = i
    return out


def _bincount_2d(ids: np.ndarray, n_ids: int) -> np.ndarray:
    """Return [n_ids] count of each id in ids."""
    flat = ids.ravel()
    return np.bincount(flat, minlength=n_ids)


def _merge_small_segments(
    ids: np.ndarray,
    min_area: int,
    max_iters: int = 4,
) -> np.ndarray:
    """Absorb segments smaller than `min_area` into their dominant neighbor.

    Uses a single-pass approximation per iteration: for every segment below
    threshold, find the most-common neighbor id along its 4-connectivity
    boundary and relabel the segment. Repeat up to `max_iters` times.
    Background id 0 is never absorbed and is never selected as a target.
    """
    H, W = ids.shape
    for _ in range(max_iters):
        n_ids = int(ids.max()) + 1
        if n_ids <= 1:
            break
        counts = _bincount_2d(ids, n_ids)
        small = np.where((counts < min_area) & (np.arange(n_ids) != 0))[0]
        if small.size == 0:
            break

        # Build a "is this pixel a member of any small segment" mask.
        small_set = np.zeros(n_ids, dtype=bool)
        small_set[small] = True
        is_small = small_set[ids]

        # Dilate the small-segment regions by 1 pixel; the new pixels are
        # neighbor candidates. We then read the neighbor id at every dilated
        # pixel and accumulate (small_id, neighbor_id) histogram.
        kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        dilated = cv2.dilate(is_small.astype(np.uint8), kernel) > 0
        # Boundary pixels: dilated AND not small.
        boundary = dilated & ~is_small
        if not boundary.any():
            # No exterior boundary to merge into; relabel to background.
            new_ids = ids.copy()
            new_ids[is_small] = 0
            ids = new_ids
            continue

        # For each small segment, gather the modal neighbor id.
        # Approach: pair each boundary pixel with its closest small neighbor.
        # We use cv2.dilate again to propagate small ids into the boundary band.
        small_id_map = np.where(is_small, ids, 0).astype(np.int32)
        # Dilate small ids by 1 — picks up the small id touching each boundary pixel.
        small_id_in_boundary = cv2.dilate(small_id_map.astype(np.uint16), kernel)
        # Build (small_id, neighbor_id) → count
        sid = small_id_in_boundary[boundary].astype(np.int64)
        nid = ids[boundary].astype(np.int64)
        flat_key = sid * n_ids + nid
        hist = np.bincount(flat_key, minlength=n_ids * n_ids).reshape(n_ids, n_ids)
        # Don't allow background (id 0) as merge target if a non-bg neighbor is available.
        # Achieved by dampening column 0 to -1; if it's the only column with mass, keep 0.
        hist_no_bg = hist.copy()
        hist_no_bg[:, 0] = 0
        nonzero_targets = hist_no_bg.sum(axis=1) > 0
        # For each small segment, pick argmax(hist) preferring non-bg.
        best_target = np.where(nonzero_targets, hist_no_bg.argmax(axis=1), hist.argmax(axis=1))

        # Build a LUT id -> new id (identity except for small segments).
        lut = np.arange(n_ids, dtype=np.int64)
        lut[small] = best_target[small]
        # Resolve transitive remaps (a small -> small chain): fixed-point iterate.
        for _i in range(4):
            new_lut = lut[lut]
            if np.array_equal(new_lut, lut):
                break
            lut = new_lut
        ids = lut[ids]

    # Renumber descending area.
    n_ids = int(ids.max()) + 1
    counts = _bincount_2d(ids, n_ids)
    # Stable argsort by (-count); keep 0 anchored at 0 if present.
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(n_ids, dtype=np.int64)
    rank[order] = np.arange(n_ids)
    return rank[ids].astype(np.int32)


def fuse_equivalence_classes(
    per_view_maps: np.ndarray,
    min_area: int = 64,
) -> np.ndarray:
    """Equivalence-class fusion of V per-view uint16 id maps.

    Args:
        per_view_maps: [V, H, W] uint16. id 0 = background per view.
        min_area: tiny segments below this pixel count are merged into the
            dominant 4-connectivity neighbor.

    Returns:
        raw_ids: [H, W] int32. 0 is reserved for the background sink (used
            by the merge step); strictly positive ids are real segments,
            sorted by descending area.
    """
    if per_view_maps.ndim != 3:
        raise ValueError(f"per_view_maps must be [V,H,W], got {per_view_maps.shape}")
    V, H, W = per_view_maps.shape
    if per_view_maps.dtype != np.uint16:
        per_view_maps = per_view_maps.astype(np.uint16)

    # Build per-pixel V-tuple signature via a contiguous void view. Each
    # pixel's V*2 bytes become a single hashable record; np.unique groups
    # exact duplicates.
    pv = np.ascontiguousarray(per_view_maps.transpose(1, 2, 0))  # [H, W, V]
    sig = pv.reshape(-1, V).view(np.dtype((np.void, V * 2))).reshape(-1)
    _uniq, inverse = np.unique(sig, return_inverse=True)
    inverse = inverse.reshape(H, W).astype(np.int64)

    # Descending-area renumber.
    n_ids = int(inverse.max()) + 1
    counts = _bincount_2d(inverse, n_ids)
    order = np.argsort(-counts, kind="stable")
    rank = np.empty(n_ids, dtype=np.int64)
    rank[order] = np.arange(n_ids)
    raw_ids = rank[inverse].astype(np.int32)

    # Merge sub-min_area segments.
    if min_area > 0:
        raw_ids = _merge_small_segments(raw_ids, min_area=int(min_area))
    if int(raw_ids.max()) > 65535:
        raise ValueError(
            f"raw_ids max {int(raw_ids.max())} exceeds uint16 — downstream "
            "uint16 PNG / npy storage will overflow. Tune --min_area up."
        )
    return raw_ids


def compute_segment_features(
    raw_ids: np.ndarray,
    rgb_uint8: np.ndarray,
) -> np.ndarray:
    """Per-segment 6-D feature vector used by global K-means.

    Features (all view-symmetric so they apply equally at training time
    on the reference direction and at inference time on a single input):

        f[0], f[1]  Lab a*, Lab b* mean within segment   (chromaticity)
        f[2], f[3]  cx, cy   centroid normalized to [0, 1]
        f[4]        log10(area / total_pixels + 1e-8)    (log-area-fraction)
        f[5]        L* mean / 100                        (luminance)

    Args:
        raw_ids: [H, W] int32 segment id map (0 included as a real bucket).
        rgb_uint8: [H, W, 3] uint8 RGB reference image.

    Returns:
        features: [N, N_FEATURES] float32 where N = raw_ids.max() + 1.
    """
    if raw_ids.ndim != 2:
        raise ValueError(f"raw_ids must be [H,W], got {raw_ids.shape}")
    if rgb_uint8.ndim != 3 or rgb_uint8.shape[-1] != 3:
        raise ValueError(f"rgb_uint8 must be [H,W,3], got {rgb_uint8.shape}")
    if rgb_uint8.dtype != np.uint8:
        rgb_uint8 = np.clip(rgb_uint8, 0, 255).astype(np.uint8)
    H, W = raw_ids.shape

    n_ids = int(raw_ids.max()) + 1
    flat = raw_ids.reshape(-1)
    total = float(H * W)

    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2Lab).astype(np.float32)
    L = lab[..., 0].reshape(-1)
    a = lab[..., 1].reshape(-1) - 128.0  # center a*
    b = lab[..., 2].reshape(-1) - 128.0  # center b*

    counts = np.bincount(flat, minlength=n_ids).astype(np.float64)
    counts_safe = np.maximum(counts, 1.0)
    sum_L = np.bincount(flat, weights=L, minlength=n_ids)
    sum_a = np.bincount(flat, weights=a, minlength=n_ids)
    sum_b = np.bincount(flat, weights=b, minlength=n_ids)

    ys, xs = np.indices((H, W))
    sum_x = np.bincount(flat, weights=xs.reshape(-1).astype(np.float64), minlength=n_ids)
    sum_y = np.bincount(flat, weights=ys.reshape(-1).astype(np.float64), minlength=n_ids)

    mean_a = (sum_a / counts_safe).astype(np.float32)
    mean_b = (sum_b / counts_safe).astype(np.float32)
    mean_x = (sum_x / counts_safe / max(W - 1, 1)).astype(np.float32)
    mean_y = (sum_y / counts_safe / max(H - 1, 1)).astype(np.float32)
    log_area = np.log10(counts / total + 1e-8).astype(np.float32)
    mean_L = (sum_L / counts_safe / 100.0).astype(np.float32)

    features = np.stack([mean_a, mean_b, mean_x, mean_y, log_area, mean_L], axis=1)
    return features.astype(np.float32)


def assign_to_centroids(
    features: np.ndarray,
    centroids: np.ndarray,
) -> np.ndarray:
    """Nearest-centroid assignment in Euclidean feature space.

    Args:
        features:   [N, D] float32
        centroids:  [K, D] float32

    Returns:
        labels: [N] int64 in [0, K).
    """
    if features.size == 0:
        return np.zeros(0, dtype=np.int64)
    if features.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"feature dim {features.shape[1]} != centroid dim {centroids.shape[1]}"
        )
    # ||f - c||^2 = ||f||^2 - 2 f.c + ||c||^2; the ||f||^2 term is constant
    # per row so we can skip it for argmin purposes.
    fc = features @ centroids.T  # [N, K]
    c_sq = (centroids ** 2).sum(axis=1)  # [K]
    # We want argmin of -2 fc + c_sq.
    scores = -2.0 * fc + c_sq[None, :]
    return np.argmin(scores, axis=1).astype(np.int64)


def rasterize_segment_labels(
    raw_ids: np.ndarray,
    labels: np.ndarray,
    *,
    background_id: int = 0,
    id_offset: int = 1,
) -> np.ndarray:
    """Map per-segment K-means labels back to a per-pixel uint16 image.

    Args:
        raw_ids: [H, W] int32 from `fuse_equivalence_classes`.
        labels:  [N] int with N = raw_ids.max() + 1, values in [0, K).
        background_id: which raw id is treated as background and emitted
            as 0 unconditionally. Defaults to 0 (matches the equivalence-
            class merge sink).
        id_offset: added to non-background labels so the final image keeps
            0 reserved for background. Set to 1 (default) → fused ids
            occupy [1, K]. Set to 0 to overlap with background.

    Returns:
        sam_mip2: [H, W] uint16.
    """
    if labels.shape[0] != int(raw_ids.max()) + 1:
        raise ValueError(
            f"labels length {labels.shape[0]} must equal raw_ids.max()+1 "
            f"= {int(raw_ids.max())+1}"
        )
    lut = labels.astype(np.int32) + id_offset
    lut[background_id] = 0  # preserve background id
    out = lut[raw_ids]
    if out.max() > 65535:
        raise ValueError(
            f"K_fusion + id_offset exceeds uint16 ({int(out.max())} > 65535)"
        )
    return out.astype(np.uint16)


def fuse_scene(
    per_view_maps: np.ndarray,
    rgb_uint8: np.ndarray,
    centroids: np.ndarray | None,
    min_area: int = 64,
    id_offset: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Stage A (+ Stage C if centroids supplied).

    Args:
        per_view_maps: [V, H, W] uint16 per-view AMG id maps.
        rgb_uint8:     [H, W, 3] uint8 reference image for features.
        centroids:     [K, N_FEATURES] float32 — if not None, also project
                       to K-means ids and return the final uint16 map.
        min_area:      tiny-segment merge floor in pixels.
        id_offset:     added to K-means labels so 0 stays reserved for bg.

    Returns:
        sam_map:  [H, W] uint16. If centroids is None, this is the raw
                  equivalence-class id map (values in [0, N_raw]) clipped
                  to uint16. Otherwise it is in [0, K + id_offset].
        raw_ids:  [H, W] int32 equivalence-class ids (Stage A output).
        features: [N_raw, N_FEATURES] float32 per-segment features.
    """
    raw_ids = fuse_equivalence_classes(per_view_maps, min_area=min_area)
    features = compute_segment_features(raw_ids, rgb_uint8)
    if centroids is None:
        # Clip to uint16 for disk write. Real raw counts are usually << 65535.
        sam_map = np.clip(raw_ids, 0, 65535).astype(np.uint16)
        return sam_map, raw_ids, features
    labels = assign_to_centroids(features, centroids)
    sam_map = rasterize_segment_labels(raw_ids, labels, id_offset=id_offset)
    return sam_map, raw_ids, features


def write_id_map_npy(path: Path | str, id_map: np.ndarray) -> Path:
    """Write the production id map as a uint16 ``.npy`` blob.

    Faster than the old PNG path (no libpng encode/decode) and removes
    the "all-black PNG" failure mode entirely. Always drops a sibling
    ``*_viz.png`` uint8 RGB colorization for human inspection. The
    optional round-trip readback (gated by `set_verify_png_write(True)`)
    catches storage corruption at the cost of doubling I/O; default off
    so bulk precompute is fast.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if id_map.dtype != np.uint16:
        if id_map.min() < 0 or id_map.max() > 65535:
            raise ValueError(
                f"id_map out of uint16 range: min={int(id_map.min())} "
                f"max={int(id_map.max())}"
            )
        id_map = id_map.astype(np.uint16)
    np.save(str(path), id_map)
    if _VERIFY_PNG_WRITE:
        back = np.load(str(path))
        if back.shape != id_map.shape or back.dtype != np.uint16:
            raise RuntimeError(
                f"NPY round-trip mismatch for {path}: wrote {id_map.shape} {id_map.dtype}, "
                f"read {back.shape} {back.dtype}"
            )
        if int(back.max()) != int(id_map.max()) or int(back.min()) != int(id_map.min()):
            raise RuntimeError(
                f"NPY round-trip value-range mismatch for {path}: "
                f"wrote min={int(id_map.min())} max={int(id_map.max())}, "
                f"read min={int(back.min())} max={int(back.max())}"
            )
    write_id_map_viz(path.with_name(path.stem + "_viz.png"), id_map)
    return path


def write_id_map_debug_png(path: Path | str, id_map: np.ndarray) -> Path:
    """Debug-only 8-bit auto-stretched grayscale of an id map.

    The lossless source of truth is `sam_raw_ids.npy`. This PNG exists
    only so a human can open it in any viewer and see segment structure
    at a glance (uint16 grayscale renders near-black on a 0..65535
    display range when ids sit in [0, ~500]). Also drops a sibling
    `*_viz.png` colorized RGB.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = max(int(id_map.max()), 1)
    stretched = (id_map.astype(np.float32) / n * 255.0).clip(0, 255).astype(np.uint8)
    ok = cv2.imwrite(str(path), stretched, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {path}")
    write_id_map_viz(path.with_name(path.stem + "_viz.png"), id_map)
    return path


def write_id_map_viz(path: Path | str, id_map: np.ndarray) -> Path:
    """Save a human-viewable uint8 RGB colorization of an id map.

    Each id gets a stable random color (seed 42 so the same id is the
    same color across scenes / runs). Pixels with id 0 — the equivalence-
    class background sink (no SAM2 mask covered that pixel in any view)
    — are filled in with their nearest labeled neighbor's color so the
    viz is a clean mosaic without distracting black holes. This only
    affects the saved PNG; the on-disk id map and training data are
    untouched.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_ids = int(id_map.max()) + 1
    rng = np.random.default_rng(42)
    palette = (rng.uniform(40, 255, size=(max(n_ids, 1), 3))).astype(np.uint8)
    palette[0] = (0, 0, 0)
    ids = id_map.astype(np.int64).clip(0, n_ids - 1)

    # Nearest-neighbor fill of id==0 pixels for a clean viz mosaic.
    bg_mask = ids == 0
    if bg_mask.any() and (~bg_mask).any():
        try:
            from scipy.ndimage import distance_transform_edt

            _, (yy, xx) = distance_transform_edt(bg_mask, return_indices=True)
            ids = ids[yy, xx]
        except ImportError:
            # SciPy missing -> leave id 0 as black; better than crashing
            # the whole precompute on a viz-only convenience helper.
            pass

    rgb = palette[ids]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION])
    return path


def save_raw_artifacts(
    out_dir: Path | str,
    raw_ids: np.ndarray,
    features: np.ndarray,
    save_debug_pngs: bool = False,
) -> tuple[Path, Path]:
    """Write Stage A artifacts to disk.

    Always written (source of truth for Stage C):
    - ``sam_raw_ids.npy``      uint16 [H, W]
    - ``sam_raw_features.npy`` float32 [N, 6]

    Optional (``save_debug_pngs=True``):
    - ``sam_raw_mip2.png``     uint8 auto-stretched grayscale (debug-only)
    - ``sam_raw_mip2_viz.png`` uint8 RGB false-color (debug-only)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_clip = np.clip(raw_ids, 0, 65535).astype(np.uint16)
    raw_ids_npy = out_dir / "sam_raw_ids.npy"
    feat_npy = out_dir / "sam_raw_features.npy"
    np.save(str(feat_npy), features.astype(np.float32))
    np.save(str(raw_ids_npy), raw_clip)
    if save_debug_pngs:
        write_id_map_debug_png(out_dir / "sam_raw_mip2.png", raw_clip)
    return raw_ids_npy, feat_npy


def load_raw_artifacts(
    in_dir: Path | str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read Stage A artifacts. Returns (raw_ids int32 [H,W], features float32 [N, D])."""
    in_dir = Path(in_dir)
    raw_ids = np.load(str(in_dir / "sam_raw_ids.npy"))
    features = np.load(str(in_dir / "sam_raw_features.npy"))
    return raw_ids.astype(np.int32), features.astype(np.float32)
