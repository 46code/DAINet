"""One-shot: global K-means over all training-scene SAM2 fusion features.

After Stage A (per-scene multi-view equivalence-class fusion via
``scripts/precompute_sam.py --stage raw``), every training scene has a
``sam_raw_features.npy`` file holding `[N, 6]` per-segment features:
(Lab a*, Lab b*, cx, cy, log-area-fraction, L*-mean / 100).

This script concatenates all those features and runs MiniBatchKMeans to
learn a single set of K_fusion centroids that lives in
``data/raw/mit_mi/sam_fusion_centroids.npy`` and is consumed by both:

  - ``scripts/precompute_sam.py --stage final`` (per-scene rasterization)
  - ``data.segmentation.segment_image`` (inference-time projection)

Design detail — material-anchored initialization. When
``materials_mip2.png`` is available for the training scenes, we seed the
first K_material centroids with per-material feature means (computed from
each segment's majority-vote material class). The remaining
``K_fusion - K_material`` centroids are seeded by K-means++ on the
remaining samples. This biases the fused id space toward material-aligned
discretizations so the SegmentationEncoder's FiLM, the material head's
logits, and the per-material reflectance loss all reinforce each other.

Usage:

    python scripts/build_sam_fusion_centroids.py \\
        --raw_root data/raw/mit_mi/sam_masks \\
        --jpg_root data/raw/mit_mi/jpg \\
        --splits   train \\
        --taxonomy data/raw/mit_mi/material_taxonomy.json \\
        --k_fusion 64 \\
        --out      data/raw/mit_mi/sam_fusion_centroids.npy
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.material_io import IGNORE_INDEX, load_taxonomy, material_dominant_id  # noqa: E402
from data.sam_fusion import N_FEATURES  # noqa: E402


def _load_material_mask_raw(scene_dir: Path) -> np.ndarray | None:
    p = scene_dir / "materials_mip2.png"
    if not p.exists():
        return None
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.uint8)


def _build_lut(raw_to_contig: dict[int, int]) -> np.ndarray:
    """Materials taxonomy LUT — built once per run, reused across scenes."""
    lut = np.full(256, IGNORE_INDEX, dtype=np.int32)
    for r, c in raw_to_contig.items():
        lut[int(r)] = int(c)
    return lut


def gather_features(
    raw_root: Path,
    jpg_root: Path | None,
    splits: list[str],
    taxonomy: dict | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Walk Stage A artifacts and concatenate features.

    Returns:
        all_features:  [N_total, N_FEATURES] float32
        all_material:  [N_total] int32 (per-segment dominant material class
            using the taxonomy LUT; IGNORE_INDEX where no material map is
            available or no real class wins the majority).
    """
    feat_blocks: list[np.ndarray] = []
    mat_blocks: list[np.ndarray] = []
    lut = _build_lut(taxonomy["raw_to_contiguous"]) if taxonomy is not None else None
    missing_artifacts = 0
    shape_mismatches = 0
    for split in splits:
        split_dir = raw_root / split
        if not split_dir.exists():
            print(f"[centroids] skip missing split: {split_dir}", file=sys.stderr)
            continue
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        for scene in tqdm(
            scenes, desc=f"[gather] {split}", unit="scene", dynamic_ncols=True
        ):
            feat_path = scene / "sam_raw_features.npy"
            ids_path = scene / "sam_raw_ids.npy"
            if not feat_path.exists() or not ids_path.exists():
                missing_artifacts += 1
                continue
            features = np.load(str(feat_path)).astype(np.float32)
            n_raw = features.shape[0]
            mat_class = np.full(n_raw, IGNORE_INDEX, dtype=np.int32)
            if lut is not None and jpg_root is not None:
                raw_ids = np.load(str(ids_path)).astype(np.int32)
                mat_raw = _load_material_mask_raw(jpg_root / split / scene.name)
                if mat_raw is not None:
                    if mat_raw.shape != raw_ids.shape:
                        # Silent in the original; surface this so a stale
                        # materials cache doesn't quietly degrade the
                        # material-anchored init.
                        tqdm.write(
                            f"[centroids] {split}/{scene.name}: material mask "
                            f"shape {mat_raw.shape} != raw_ids {raw_ids.shape}; "
                            "falling back to IGNORE for this scene"
                        )
                        shape_mismatches += 1
                    else:
                        mat_remap = lut[mat_raw]
                        mat_class = material_dominant_id(mat_remap, raw_ids, n_raw)
            feat_blocks.append(features)
            mat_blocks.append(mat_class)
    if missing_artifacts:
        print(
            f"[centroids] {missing_artifacts} scenes missing Stage A artifacts "
            "(run scripts/precompute_sam.py --stage raw first)",
            file=sys.stderr,
        )
    if shape_mismatches:
        print(
            f"[centroids] {shape_mismatches} scenes had material/raw shape "
            "mismatches — material-anchored init will be partial",
            file=sys.stderr,
        )
    if not feat_blocks:
        return np.zeros((0, N_FEATURES), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.concatenate(feat_blocks, axis=0), np.concatenate(mat_blocks, axis=0)


def material_anchored_init(
    features: np.ndarray,
    material: np.ndarray,
    k_fusion: int,
    k_material: int,
    seed: int = 0,
) -> np.ndarray:
    """Initial centroid matrix.

    First `K_material` rows are per-material feature means. Remaining
    `K_fusion - K_material` rows come from K-means++ over the rest of the
    features so the seed remains spread out across the feature manifold.
    """
    K = k_fusion
    D = features.shape[1]
    init = np.zeros((K, D), dtype=np.float32)
    used = min(k_material, K)
    for c in range(used):
        mask = material == c
        if mask.any():
            init[c] = features[mask].mean(axis=0)
        else:
            # Empty class — fall back to a random sample. K-means++ refinement
            # will reshape these regardless.
            rng = np.random.default_rng(seed + c)
            idx = int(rng.integers(0, features.shape[0]))
            init[c] = features[idx]
    if used < K:
        # K-means++ seeding for the rest.
        try:
            from sklearn.cluster import kmeans_plusplus

            extra, _ = kmeans_plusplus(features, n_clusters=K - used, random_state=seed)
        except Exception:
            rng = np.random.default_rng(seed + 999)
            idx = rng.choice(features.shape[0], size=K - used, replace=False)
            extra = features[idx]
        init[used:] = extra.astype(np.float32)
    return init


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_root",
        required=True,
        help="root of Stage A artifacts (matches --out from precompute_sam --stage raw)",
    )
    parser.add_argument(
        "--jpg_root",
        default="data/raw/mit_mi/jpg",
        help="root of jpg/ (only used to locate materials_mip2.png for anchoring)",
    )
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--taxonomy",
        default="data/raw/mit_mi/material_taxonomy.json",
        help="material taxonomy JSON for the anchored init; pass '' to disable",
    )
    parser.add_argument("--k_fusion", type=int, default=64)
    parser.add_argument("--out", default="data/raw/mit_mi/sam_fusion_centroids.npy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--batch_size", type=int, default=4096, help="MiniBatchKMeans batch size"
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    taxonomy = None
    k_material = 0
    if args.taxonomy:
        try:
            taxonomy = load_taxonomy(args.taxonomy)
            k_material = int(taxonomy["K"])
        except FileNotFoundError as e:
            print(f"[centroids] {e}", file=sys.stderr)
            print("[centroids] proceeding without material-anchored init", file=sys.stderr)

    features, material = gather_features(
        Path(args.raw_root),
        Path(args.jpg_root) if args.jpg_root else None,
        splits,
        taxonomy,
    )
    if features.shape[0] == 0:
        print(
            f"[centroids] no features found under {args.raw_root}/{{{','.join(splits)}}}",
            file=sys.stderr,
        )
        return 2
    print(
        f"[centroids] gathered {features.shape[0]} segments across "
        f"{features.shape[1]}-D feature space; k_material={k_material}, "
        f"k_fusion={args.k_fusion}"
    )

    try:
        from sklearn.cluster import MiniBatchKMeans
    except Exception as exc:
        print(f"[centroids] sklearn import failed: {exc}", file=sys.stderr)
        return 2

    init = material_anchored_init(features, material, args.k_fusion, k_material, seed=args.seed)
    km = MiniBatchKMeans(
        n_clusters=args.k_fusion,
        init=init,
        n_init=1,
        random_state=args.seed,
        batch_size=args.batch_size,
        max_iter=200,
    )
    t0 = time.perf_counter()
    km.fit(features)
    fit_s = time.perf_counter() - t0
    centroids = km.cluster_centers_.astype(np.float32)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), centroids)
    print(
        f"[centroids] wrote {out_path} shape={centroids.shape} "
        f"fit={fit_s:.1f}s n_iter={int(km.n_iter_)} inertia={km.inertia_:.3g}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
