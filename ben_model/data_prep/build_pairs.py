"""Materialize MIT-MI training pairs into each baseline's expected layout.

MIT-MI train: 985 scenes x 25 directional inputs, each paired with the scene's
single ``target_clean.jpg``. We:

  1. split scenes into train / val (val_frac, seeded);
  2. apply dainet's own probe-masking protocol so the baselines train and validate
     exactly like dainet (benchmark fairness):
       - TRAIN pairs are left RAW — the model sees the chrome+gray probe spheres,
         just as dainet's training inputs are never blackened;
       - VAL pairs blacken the probe regions (from meta.json) in BOTH inputs and
         GTs, so the in-training validation metric reflects scene content only
         (dainet masks its val/test *metrics* the same way).
     Each masked/raw image is written once to a shared cache. (The held-out
     *test* set is masked at metric time by the scorer, not here.)
  3. build each repo's directory layout as *symlinks* into that cache, so the
     four layouts cost almost no extra disk.

Layouts (selectable via --layouts):
  basicsr   Restormer + Retinexformer (BasicSR Dataset_PairedImage):
              <out>/basicsr/<train|val>/{lq,gt}/<key>_d<id>.png
  rln2      RLN2 (BasicSR Dataset_PairedScene):
              <out>/rln2/<Train|Validation>/{GT/<key>_GT.png,
                IN_CR/<key>/dir_<id>.png, IN_SH/<key>/<key>_1_IN.png (+_1_SH_IN val)}
  ifblend   IFBlend (ImageSet; dir name ends ASRD6K):
              <out>/ifblend_ASRD6K/<Train|Validation>/<key>_d<id>_{in,gt}.png
  hvi       HVI-CIDNet (LOL low/high folders):
              <out>/hvi/<train|eval>/{low,high}/<key>_d<id>.png

A manifest.json records the scene<->key map and the split.

Usage:
  python build_pairs.py --scenes 0              # all 985 scenes (full run)
  python build_pairs.py --scenes 60 --layouts basicsr,rln2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from lib import config, probe_mask  # noqa: E402

N_DIR = config.N_DIRECTIONS
ALL_LAYOUTS = ["basicsr", "rln2", "ifblend", "hvi"]


def _list_train_scenes() -> list[str]:
    return sorted(p.name for p in config.MITMI_TRAIN_IN.iterdir() if p.is_dir())


def _write_image(src: Path, mask: np.ndarray | None, dst: Path) -> bool:
    """Read src (BGR) and write PNG to dst. When ``mask`` is given (val split)
    the probe pixels are zeroed; ``mask=None`` writes the image unchanged (train
    split — the model sees the probes, as in dainet training). Returns success."""
    bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if bgr is None:
        return False
    if mask is not None:
        if mask.shape[:2] != bgr.shape[:2]:
            m = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            m = mask
        bgr = (bgr.astype(np.float32) * m[..., None]).astype(np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dst), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]))


def _link(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target.resolve())


def build(scenes_n: int, val_frac: float, layouts: list[str], out: Path,
          seed: int = 42) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    cache_in = out / "_cache" / "in"
    cache_gt = out / "_cache" / "gt"

    scenes = _list_train_scenes()
    if scenes_n and scenes_n > 0:
        scenes = scenes[:scenes_n]
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(len(scenes) * val_frac)) if len(scenes) > 1 else 0
    val_scenes = set(scenes[:n_val])
    key_of = {s: f"s{ i:04d}" for i, s in enumerate(sorted(scenes))}
    manifest = {"train": [], "val": [], "key_of": key_of, "n_dir": N_DIR}

    # ---- Stage 1: materialize the per-scene cache (each image written once) ----
    # Probe-masking follows dainet's protocol: TRAIN stays raw (model sees probes);
    # VAL blackens the probe regions so the in-training val metric is clean.
    cached: dict[str, dict] = {}  # key -> {"gt": Path, "dirs": {id: Path}, "split": str}
    for scene in tqdm(sorted(scenes), desc="materialize cache", unit="scene"):
        key = key_of[scene]
        split = "val" if scene in val_scenes else "train"
        manifest[split].append({"scene": scene, "key": key})
        meta = config.MITMI_TRAIN_IN / scene / "meta.json"
        gt_src = config.MITMI_TRAIN_GT / scene / "target_clean.jpg"
        if not gt_src.exists():
            continue
        # val is probe-masked (at the GT's native resolution); train is raw
        mask = None
        if split == "val":
            probe = cv2.imread(str(gt_src), cv2.IMREAD_COLOR)
            if probe is None:
                continue
            mask = probe_mask.mitmi_probe_mask(meta, probe.shape[:2])
        gt_dst = cache_gt / f"{key}.png"
        if not gt_dst.exists():
            _write_image(gt_src, mask, gt_dst)
        dirs = {}
        for did in range(N_DIR):
            src = config.MITMI_TRAIN_IN / scene / f"dir_{did}_mip2.jpg"
            if not src.exists():
                continue
            dst = cache_in / f"{key}_d{did}.png"
            if not dst.exists():
                if not _write_image(src, mask, dst):
                    continue
            dirs[did] = dst
        cached[key] = {"gt": gt_dst, "dirs": dirs, "split": split}

    # ---- Stage 2: build each layout from the cache via symlinks ----
    for key, info in tqdm(cached.items(), desc="link layouts", unit="scene"):
        gt, dirs, split = info["gt"], info["dirs"], info["split"]
        if not dirs:
            continue
        if "basicsr" in layouts:
            sp = "train" if split == "train" else "val"
            for did, ip in dirs.items():
                _link(ip, out / "basicsr" / sp / "lq" / f"{key}_d{did}.png")
                _link(gt, out / "basicsr" / sp / "gt" / f"{key}_d{did}.png")
        if "rln2" in layouts:
            ph = "Train" if split == "train" else "Validation"
            _link(gt, out / "rln2" / ph / "GT" / f"{key}_GT.png")
            for did, ip in dirs.items():
                _link(ip, out / "rln2" / ph / "IN_CR" / key / f"dir_{did}.png")
            rep = dirs[min(dirs)]
            _link(rep, out / "rln2" / ph / "IN_SH" / key / f"{key}_1_IN.png")
            _link(rep, out / "rln2" / ph / "IN_SH" / key / f"{key}_1_SH_IN.png")
        if "ifblend" in layouts:
            st = "Train" if split == "train" else "Validation"
            for did, ip in dirs.items():
                _link(ip, out / "ifblend_ASRD6K" / st / f"{key}_d{did}_in.png")
                _link(gt, out / "ifblend_ASRD6K" / st / f"{key}_d{did}_gt.png")
        if "hvi" in layouts:
            sp = "train" if split == "train" else "eval"
            for did, ip in dirs.items():
                _link(ip, out / "hvi" / sp / "low" / f"{key}_d{did}.png")
                _link(gt, out / "hvi" / sp / "high" / f"{key}_d{did}.png")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_pairs = sum(len(v["dirs"]) for v in cached.values())
    summary = {"n_scenes": len(cached), "n_train_scenes": len(manifest["train"]),
               "n_val_scenes": len(manifest["val"]), "n_pairs": n_pairs,
               "layouts": layouts, "out": str(out)}
    print(f"[build_pairs] {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", type=int, default=0, help="cap scene count; 0 = all 985")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--layouts", default=",".join(ALL_LAYOUTS))
    ap.add_argument("--out", default=str(config.DATA_PREP_OUT))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    build(args.scenes, args.val_frac, layouts, Path(args.out), seed=args.seed)


if __name__ == "__main__":
    main()
