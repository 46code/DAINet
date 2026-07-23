"""Two-stage SAM2 multi-view fusion precompute.

The old single-direction (`dir_0_mip2.jpg`) precompute is gone. We now fuse
SAM2 AMG outputs from all V (≈25) directional views of every scene and then
project per-segment features into a globally-shared K-means id space so the
SegmentationEncoder sees consistent ids across scenes and at inference.

Stages (run in order):

    1. ``--stage raw`` runs SAM2 AMG on each ``dir_{0..n_views-1}_mip2.jpg``,
       fuses to equivalence-class raw segments, computes per-segment 6-D
       features, and writes the following per-scene:

           <out_root>/{split}/<scene>/sam_raw_ids.npy        # [H, W] uint16 (source of truth)
           <out_root>/{split}/<scene>/sam_raw_features.npy   # [N, 6]  float32

       Optional with ``--save-debug-pngs``:
           <out_root>/{split}/<scene>/sam_raw_mip2.png       # uint8 grayscale (debug)
           <out_root>/{split}/<scene>/sam_raw_mip2_viz.png   # uint8 RGB (debug)

    2. ``scripts/build_sam_fusion_centroids.py`` (separate command) reads
       every training scene's features, runs material-anchored
       MiniBatchKMeans, and writes ``data/sam_fusion_centroids.npy``.

    3. ``--stage final`` reads back the raw artifacts + the global
       centroids, assigns each segment to its nearest centroid, and
       rasterizes to the on-disk contract:

           <out_root>/{split}/<scene>/sam_mip2.npy           # [H, W] uint16, ids in [0, K_fusion]
           <out_root>/{split}/<scene>/sam_mip2_viz.png       # uint8 RGB (debug-only)

The dataset (`data/segmentation.load_sam_mask` -> `int32 [H, W]`) reads
``sam_mip2.npy`` directly — uncompressed numpy load is ~10x faster than
PNG decode and removes the "all-black PNG" failure mode. Inference re-uses
the same centroids via ``data.segmentation.segment_image`` so the id space
at deploy time matches the id space at training time.

Install (SAM2 only, no SAM3 — SAM3 has no AutomaticMaskGenerator):

    git clone https://github.com/facebookresearch/sam2 ~/my_model/sam2
    pip install -e ~/my_model/sam2

Usage:

    # Stage A — multi-view AMG + equivalence-class fusion. Run once.
    python scripts/precompute_sam.py --stage raw \\
        --root data/raw/mit_mi/jpg \\
        --splits train,test \\
        --out  data/raw/mit_mi/sam_masks \\
        --weights ~/my_model/sam2_weights/sam2.1_hiera_large.pt \\
        --gpu 0

    # Stage B — global K-means (see other script). Run once.
    python scripts/build_sam_fusion_centroids.py \\
        --raw_root data/raw/mit_mi/sam_masks \\
        --jpg_root data/raw/mit_mi/jpg \\
        --splits train --k_fusion 64 \\
        --out data/sam_fusion_centroids.npy

    # Stage C — per-scene rasterization. Run once.
    python scripts/precompute_sam.py --stage final \\
        --out  data/raw/mit_mi/sam_masks \\
        --splits train,test \\
        --centroids data/sam_fusion_centroids.npy
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Local imports — keep them at module scope so the script doubles as a
# thin entry point on `python -m scripts.precompute_sam`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.sam_fusion import (  # noqa: E402
    assign_to_centroids,
    fuse_equivalence_classes,
    compute_segment_features,
    per_view_id_map,
    rasterize_segment_labels,
    save_raw_artifacts,
    load_raw_artifacts,
    write_id_map_npy,
    set_verify_png_write,
)


DEFAULT_WEIGHTS = "~/my_model/sam2_weights/sam2.1_hiera_large.pt"
DEFAULT_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"


def _try_load_sam2(weights_path: Path, config: str, device: str, points_per_side: int):
    """Load the SAM2 automatic mask generator. Exit on failure."""
    try:
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        print(
            "[precompute_sam] SAM2 import failed. Install with:\n"
            "    git clone https://github.com/facebookresearch/sam2 ~/my_model/sam2\n"
            "    pip install -e ~/my_model/sam2\n"
            f"Underlying error: {exc}",
            file=sys.stderr,
        )
        sys.exit(2)

    weights_path = weights_path.expanduser()
    if not weights_path.exists():
        print(
            f"[precompute_sam] weights not found at {weights_path}\n"
            "Download a SAM2 checkpoint (e.g. sam2.1_hiera_large.pt) from "
            "https://github.com/facebookresearch/sam2#model-description and "
            "place it there.",
            file=sys.stderr,
        )
        sys.exit(2)

    sam = build_sam2(config, str(weights_path), device=device, apply_postprocessing=False)
    return SAM2AutomaticMaskGenerator(sam, points_per_side=points_per_side)


def _read_view_rgb(path: Path) -> np.ndarray | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _gpu_mem_mb(torch_mod) -> float:
    if torch_mod is None:
        return 0.0
    try:
        return float(torch_mod.cuda.memory_allocated()) / (1024.0 * 1024.0)
    except Exception:
        return 0.0


def _run_stage_raw(args: argparse.Namespace) -> int:
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = "cuda"
    else:
        device = "cpu"

    try:
        import torch
    except Exception:
        torch = None  # type: ignore[assignment]

    if torch is not None and device == "cuda":
        # Bit-neutral for SAM2's static-shape image encoder; lets cuDNN pick
        # the fastest conv algorithm after a short warmup.
        torch.backends.cudnn.benchmark = True

    predictor = _try_load_sam2(
        Path(args.weights).expanduser(), args.config, device, args.points_per_side
    )

    if args.compile and torch is not None and device == "cuda":
        # Compile only the image encoder. Prompt-decoder shapes vary across
        # views (point grids vs. variable mask counts), so leave it eager.
        try:
            inner = predictor.predictor.model  # SAM2AutomaticMaskGenerator -> SAM2ImagePredictor -> model
            inner.image_encoder = torch.compile(
                inner.image_encoder, mode="reduce-overhead", fullgraph=False
            )
            tqdm.write("[precompute_sam] torch.compile applied to image_encoder (reduce-overhead).")
        except Exception as exc:
            tqdm.write(f"[precompute_sam] torch.compile failed, continuing eager: {exc}")

    autocast_enabled = bool(args.fp16) and device == "cuda" and torch is not None
    autocast_dtype = torch.bfloat16 if (autocast_enabled and args.bf16) else (
        torch.float16 if autocast_enabled else None
    )
    print(
        f"[precompute_sam] stage=raw weights={args.weights} device={device} "
        f"gpu={args.gpu} points_per_side={args.points_per_side} "
        f"autocast={autocast_enabled} dtype={str(autocast_dtype).rsplit('.', 1)[-1] if autocast_dtype else 'fp32'} "
        f"compile={bool(args.compile)} verify_png={args.verify_png} "
        f"prefetch_workers={args.prefetch_workers} save_debug_pngs={bool(args.save_debug_pngs)}"
    )

    root = Path(args.root)
    out_root = Path(args.out)
    for split in args.splits.split(","):
        split = split.strip()
        split_dir = root / split
        if not split_dir.exists():
            print(f"[precompute_sam] split missing on disk: {split_dir}", file=sys.stderr)
            continue
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        scene_bar = tqdm(scenes, desc=f"[raw] {split}", unit="scene", dynamic_ncols=True)
        for scene in scene_bar:
            out_dir = out_root / split / scene.name
            raw_marker = out_dir / "sam_raw_ids.npy"
            if raw_marker.exists() and not args.overwrite:
                tqdm.write(f"[skip] {split}/{scene.name}: cached")
                continue
            view_paths = [scene / f"dir_{v}_mip2.jpg" for v in range(args.n_views)]
            existing = [(v, p) for v, p in enumerate(view_paths) if p.exists()]
            if len(existing) < max(args.min_views, 1):
                tqdm.write(
                    f"[skip] {split}/{scene.name}: only "
                    f"{len(existing)}/{args.n_views} views on disk"
                )
                continue

            # Probe shape from the first view so we can allocate per_view up
            # front; AMG sees each view in order.
            first_rgb = _read_view_rgb(existing[0][1])
            if first_rgb is None:
                tqdm.write(f"[skip] {split}/{scene.name}: first view unreadable")
                continue
            H, W = first_rgb.shape[:2]
            per_view = np.zeros((args.n_views, H, W), dtype=np.uint16)
            reference_rgb = first_rgb

            scene_t0 = time.perf_counter()
            view_times = []
            t_io_total = 0.0
            t_amg_total = 0.0
            # Prefetch view JPGs so disk reads overlap with GPU AMG.
            with ThreadPoolExecutor(max_workers=max(1, args.prefetch_workers)) as pool:
                futures: dict[int, object] = {}
                # Seed first view with the already-decoded buffer.
                futures[existing[0][0]] = pool.submit(lambda r=first_rgb: r)
                # Kick off the next prefetch (idx=1) eagerly.
                for idx, (v, p) in enumerate(existing[1 : 1 + args.prefetch_workers], start=1):
                    futures[v] = pool.submit(_read_view_rgb, p)

                view_bar = tqdm(
                    existing,
                    desc=scene.name,
                    unit="view",
                    leave=False,
                    dynamic_ncols=True,
                )
                for idx, (v, p) in enumerate(view_bar):
                    t_io_start = time.perf_counter()
                    rgb = futures.pop(v).result()
                    t_io_total += time.perf_counter() - t_io_start
                    # Queue the view (idx + 1 + prefetch_workers) reads ahead.
                    next_idx = idx + 1 + args.prefetch_workers
                    if next_idx < len(existing):
                        nv, np_path = existing[next_idx]
                        futures[nv] = pool.submit(_read_view_rgb, np_path)
                    if rgb is None:
                        continue
                    if v == 0:
                        reference_rgb = rgb
                    t0 = time.perf_counter()
                    if autocast_enabled:
                        with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                            masks = predictor.generate(rgb)
                    else:
                        masks = predictor.generate(rgb)
                    per_view[v] = per_view_id_map(masks, (H, W))
                    del masks
                    dt_amg = time.perf_counter() - t0
                    t_amg_total += dt_amg
                    view_times.append(dt_amg)
                    view_bar.set_postfix(
                        last_s=f"{dt_amg:.2f}",
                        gpu_mb=f"{_gpu_mem_mb(torch):.0f}",
                    )

            t_fuse_start = time.perf_counter()
            raw_ids = fuse_equivalence_classes(per_view, min_area=args.min_area)
            t_fuse = time.perf_counter() - t_fuse_start

            t_feat_start = time.perf_counter()
            features = compute_segment_features(raw_ids, reference_rgb)
            t_feat = time.perf_counter() - t_feat_start

            t_save_start = time.perf_counter()
            save_raw_artifacts(out_dir, raw_ids, features, save_debug_pngs=args.save_debug_pngs)
            t_save = time.perf_counter() - t_save_start

            # End-of-scene defensive flush (replaces the per-view churn).
            if torch is not None and device == "cuda":
                torch.cuda.empty_cache()

            scene_s = time.perf_counter() - scene_t0
            mean_view = sum(view_times) / max(len(view_times), 1)
            tqdm.write(
                f"[done] {split}/{scene.name}: views={len(view_times)} "
                f"raw_segs={int(raw_ids.max())+1} "
                f"scene={scene_s:.1f}s view_mean={mean_view:.2f}s "
                f"amg={t_amg_total:.1f}s io={t_io_total:.2f}s "
                f"fuse={t_fuse:.2f}s feat={t_feat:.2f}s save={t_save:.2f}s "
                f"gpu_mb={_gpu_mem_mb(torch):.0f} -> {out_dir}"
            )
            scene_bar.set_postfix(
                last_s=f"{scene_s:.1f}",
                view_s=f"{mean_view:.2f}",
                amg_s=f"{t_amg_total:.1f}",
                raw_segs=int(raw_ids.max()) + 1,
                gpu_mb=f"{_gpu_mem_mb(torch):.0f}",
            )
    return 0


def _run_stage_final(args: argparse.Namespace) -> int:
    centroids_path = Path(args.centroids).expanduser()
    if not centroids_path.exists():
        print(
            f"[precompute_sam] centroids file missing: {centroids_path}\n"
            "Run scripts/build_sam_fusion_centroids.py first.",
            file=sys.stderr,
        )
        return 2
    centroids = np.load(str(centroids_path)).astype(np.float32)
    print(
        f"[precompute_sam] stage=final centroids={centroids.shape} from {centroids_path} "
        f"verify_png={args.verify_png}"
    )

    out_root = Path(args.out)
    for split in args.splits.split(","):
        split = split.strip()
        split_dir = out_root / split
        if not split_dir.exists():
            print(f"[precompute_sam] split missing on disk: {split_dir}", file=sys.stderr)
            continue
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        scene_bar = tqdm(scenes, desc=f"[final] {split}", unit="scene", dynamic_ncols=True)
        for scene in scene_bar:
            out_dir = scene
            raw_path = out_dir / "sam_raw_ids.npy"
            feat_path = out_dir / "sam_raw_features.npy"
            if not raw_path.exists() or not feat_path.exists():
                tqdm.write(
                    f"[skip] {split}/{scene.name}: missing Stage A artifacts "
                    "(run --stage raw first)"
                )
                continue
            t0 = time.perf_counter()
            raw_ids, features = load_raw_artifacts(out_dir)
            labels = assign_to_centroids(features, centroids)
            sam_map = rasterize_segment_labels(raw_ids, labels, id_offset=args.id_offset)
            out_path = out_dir / "sam_mip2.npy"
            write_id_map_npy(out_path, sam_map)
            uniq = np.unique(sam_map)
            dt = time.perf_counter() - t0
            tqdm.write(
                f"[done] {split}/{scene.name}: max_id={int(sam_map.max())} "
                f"n_unique={uniq.size} t={dt:.2f}s -> {out_path}"
            )
            scene_bar.set_postfix(last_s=f"{dt:.2f}", n_unique=int(uniq.size))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("raw", "final"),
        required=True,
        help="raw = multi-view AMG + equivalence-class fusion; "
        "final = K-means projection to global ids",
    )
    parser.add_argument("--root", help="root of jpg/ (with train/, test/ inside) — stage raw only")
    parser.add_argument("--out", required=True, help="output root for SAM2 artifacts")
    parser.add_argument("--splits", default="train,test")
    parser.add_argument(
        "--n_views",
        type=int,
        default=25,
        help="number of per-scene directional views to consume (stage raw)",
    )
    parser.add_argument(
        "--min_views",
        type=int,
        default=20,
        help="minimum views present on disk before a scene is processed (stage raw)",
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=64,
        help="post-fusion min segment area in pixels (stage raw)",
    )
    parser.add_argument(
        "--weights",
        default=DEFAULT_WEIGHTS,
        help="path to a SAM2 .pt checkpoint (stage raw)",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="SAM2 hydra config name (stage raw)",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="CUDA device index; -1 for CPU (stage raw)",
    )
    parser.add_argument(
        "--points_per_side",
        type=int,
        default=32,
        help="AMG grid density (stage raw)",
    )
    parser.add_argument(
        "--centroids",
        default="data/sam_fusion_centroids.npy",
        help="path to global K-means centroids (stage final)",
    )
    parser.add_argument(
        "--id_offset",
        type=int,
        default=1,
        help="offset added to K-means labels so 0 stays reserved for background "
        "(stage final)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run SAM2 AMG under torch.autocast on CUDA (default: on; "
        "dtype is float16 unless --bf16 is set)",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="use torch.bfloat16 instead of float16 in the autocast block. "
        "Recommended on Ada/Hopper (RTX 6000-Ada, H100) for numerical "
        "stability with identical speed; no effect if --no-fp16 is set",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the SAM2 image encoder (reduce-overhead). "
        "Experimental: bit-neutral if it compiles, falls back to eager on failure",
    )
    parser.add_argument(
        "--prefetch_workers",
        type=int,
        default=4,
        help="background threads prefetching next view JPGs (stage raw)",
    )
    parser.add_argument(
        "--save-debug-pngs",
        dest="save_debug_pngs",
        action="store_true",
        help="also emit sam_raw_mip2.png + sam_raw_mip2_viz.png for human "
        "inspection. Off by default — the .npy files are the source of truth "
        "and downstream stages do not read the PNGs",
    )
    parser.add_argument(
        "--verify-png",
        dest="verify_png",
        action="store_true",
        help="enable the post-write readback sanity check in write_id_map_png "
        "(doubles PNG I/O; useful for one-off audits, off by default)",
    )
    args = parser.parse_args()

    set_verify_png_write(args.verify_png)

    if args.stage == "raw":
        if not args.root:
            parser.error("--root is required for --stage raw")
        return _run_stage_raw(args)
    return _run_stage_final(args)


if __name__ == "__main__":
    sys.exit(main())
