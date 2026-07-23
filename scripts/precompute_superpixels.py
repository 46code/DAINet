"""Pre-compute chromaticity K-means super-pixels for region-aware losses.

Materials_mip2.png has been removed from the pipeline (replaced by
SAM3 for FiLM conditioning). The region-aware chromatic loss
(``region_chroma_variance``) still needs a per-scene segment map whose
segments are *material-uniform* — i.e. one paint color per segment.

We obtain this **without hand labels** by K-means-clustering the GT
target image's chromaticity (CIE Lab a*, b* channels) per scene. The
result is a uint8 ``.npy`` of cluster ids saved alongside the dataset:

    <out>/{split}/<scene>/chroma_clusters_mip2.npy

NPY is preferred over PNG: uncompressed numpy load is ~10x faster than
libpng decode, the on-disk size delta is negligible for uint8 K≤8 maps,
and there is no "all-black PNG" failure mode to guard against.

L* is intentionally excluded so the clusters group by *material color*,
not by shading or shadow brightness. K=8 is enough to distinguish the
typical room palette (wall paint, floor, ceiling, woodwork, furniture
fabrics) on MIT-MI scenes.

The loss layer (`losses.region`) reads this file via
``data.segmentation.load_chroma_superpixels`` (uint8 → int32 cast) and
consumes only the *partition* — the absolute cluster ids don't matter,
only "which pixels share a segment." That means parallelizing across
scenes and seeding the cluster centers does not change the loss
signal; it only changes the wall time.

Usage:

    python scripts/precompute_superpixels.py \\
        --gt_root data/raw/mit_mi/jpg_gt \\
        --splits  train,test \\
        --out     data/raw/mit_mi/superpixels \\
        --k 8 \\
        --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


@dataclass
class _SceneTask:
    scene_path: Path
    out_dir: Path
    k: int
    seed: int
    overwrite: bool


def _process_scene(task: _SceneTask) -> tuple[str, str, int]:
    """Worker: run K-means on a single scene's GT chromaticity.

    Returns (status, scene_name, n_clusters) where status is one of
    {"done", "skip", "missing", "unreadable"}. n_clusters is 0 for any
    non-"done" status.
    """
    out_path = task.out_dir / "chroma_clusters_mip2.npy"
    if out_path.exists() and not task.overwrite:
        return ("skip", task.scene_path.name, 0)
    tgt_path = task.scene_path / "target_clean.jpg"
    if not tgt_path.exists():
        return ("missing", task.scene_path.name, 0)

    # OpenCV is already multi-threaded internally; pinning to 1 thread
    # per worker avoids oversubscription on a shared box.
    cv2.setNumThreads(1)
    cv2.setRNGSeed(task.seed)

    bgr = cv2.imread(str(tgt_path), cv2.IMREAD_COLOR)
    if bgr is None:
        return ("unreadable", task.scene_path.name, 0)

    # OpenCV's Lab conversion produces the same Lab values whether the
    # input is BGR or RGB — the channel-order argument only tells it how
    # to interpret the source. Skipping the intermediate BGR→RGB hop
    # saves an allocation per scene.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = np.stack([lab[..., 1] - 128.0, lab[..., 2] - 128.0], axis=-1)
    H, W, _ = ab.shape
    samples = ab.reshape(-1, 2).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
    _, labels, _ = cv2.kmeans(
        samples,
        K=task.k,
        bestLabels=None,
        criteria=criteria,
        attempts=3,
        flags=cv2.KMEANS_PP_CENTERS,
    )
    ids = labels.reshape(H, W).astype(np.uint8)

    task.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), ids)
    return ("done", task.scene_path.name, int(ids.max()) + 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_root", required=True, help="root of jpg_gt/ (with train/, test/)")
    parser.add_argument("--out", required=True, help="output root for super-pixel masks")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument("--k", type=int, default=8, help="number of clusters per scene")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(4, (os.cpu_count() or 4) // 2)),
        help="parallel worker processes (CPU-bound; default ~half the cores capped at 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="OpenCV RNG seed for kmeans init (deterministic cluster ids per scene)",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    gt_root = Path(args.gt_root)
    out_root = Path(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    print(
        f"[precompute_superpixels] k={args.k} workers={args.workers} seed={args.seed} "
        f"splits={splits} gt_root={gt_root} out={out_root}"
    )

    total_done = 0
    total_skip = 0
    total_t0 = time.perf_counter()
    for split in splits:
        split_dir = gt_root / split
        if not split_dir.exists():
            print(
                f"[precompute_superpixels] split missing on disk: {split_dir}",
                file=sys.stderr,
            )
            continue
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        tasks = [
            _SceneTask(
                scene_path=scene,
                out_dir=out_root / split / scene.name,
                k=args.k,
                seed=args.seed,
                overwrite=args.overwrite,
            )
            for scene in scenes
        ]

        bar = tqdm(total=len(tasks), desc=f"[chroma] {split}", unit="scene", dynamic_ncols=True)
        if args.workers <= 1:
            results_iter = (_process_scene(t) for t in tasks)
        else:
            pool = Pool(processes=args.workers)
            results_iter = pool.imap_unordered(_process_scene, tasks)

        try:
            for status, scene_name, n_clusters in results_iter:
                if status == "done":
                    total_done += 1
                    bar.set_postfix(last_scene=scene_name, n_clusters=n_clusters)
                elif status == "skip":
                    total_skip += 1
                else:
                    tqdm.write(f"[{status}] {split}/{scene_name}")
                bar.update(1)
        finally:
            bar.close()
            if args.workers > 1:
                pool.close()
                pool.join()

    dt = time.perf_counter() - total_t0
    print(
        f"[precompute_superpixels] done={total_done} skip={total_skip} "
        f"wall={dt:.1f}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
