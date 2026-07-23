"""Score one model's predictions on one dataset against the GT.

Reads predictions saved by the per-model runners at
``results/<dataset>/<model>/preds/<key>.png`` (mit_mi keys are
``<scene>/dir_<id>``), computes the ben_guide.md metric family per sample
using the shared (DAINet) backbone, and writes:

  results/<dataset>/<model>/metrics.json   (aggregates + per_sample + meta)
  results/<dataset>/<model>/per_sample.csv (one row per sample, for stats)

All metrics are probe-mask-aware on mit_mi.

Usage:
  python eval_all.py --dataset mit_mi --model restormer --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib import config, datasets, metricsx  # noqa: E402

CSV_KEYS = ["psnr", "ssim", "ms_ssim", "lpips"]


def _to_chw(arr: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)


def _find_pred(preds_dir: Path, key: str) -> Path | None:
    for ext in (".png", ".jpg"):
        p = preds_dir / f"{key}{ext}"
        if p.exists():
            return p
    return None


def score(dataset: str, model: str, device: str, limit_scenes: int = 0,
          with_lpips: bool = True, results_root: Path | None = None) -> dict:
    results_root = results_root or config.RESULTS
    out_dir = results_root / dataset / model
    preds_dir = out_dir / "preds"
    if not preds_dir.exists():
        raise FileNotFoundError(f"no predictions at {preds_dir}")

    t0 = time.time()
    per_sample: list[dict] = []
    for s in datasets.iter_samples(dataset, limit_scenes=limit_scenes):
        pp = _find_pred(preds_dir, s.key)
        if pp is None:
            continue
        pred = datasets.read_rgb01(pp)
        gt = datasets.read_rgb01(s.gt_path)
        if pred is None or gt is None:
            continue
        if pred.shape[:2] != gt.shape[:2]:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
        mask = datasets.load_mask(s, gt.shape[:2])
        pred_t = _to_chw(pred, device)
        gt_t = _to_chw(gt, device)
        mask_t = torch.from_numpy(mask)[None, None].to(device)
        row = metricsx.score_pair(pred_t, gt_t, mask=mask_t, with_lpips=with_lpips)
        row.update({"scene": s.scene, "direction_id": s.direction_id, "key": s.key})
        per_sample.append(row)

    if not per_sample:
        raise RuntimeError(f"no predictions matched GT under {preds_dir}")

    agg = metricsx.aggregate(per_sample, CSV_KEYS)
    out = {
        "method": model,
        "dataset": dataset,
        "meta": {
            "n_samples": len(per_sample),
            "runtime_sec": time.time() - t0,
            "device": device,
            "with_lpips": with_lpips,
        },
        "aggregates": agg,
        "per_sample": per_sample,
    }
    (out_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    with (out_dir / "per_sample.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "direction_id", "key"] + CSV_KEYS)
        w.writeheader()
        for r in per_sample:
            w.writerow({k: r.get(k, "") for k in (["scene", "direction_id", "key"] + CSV_KEYS)})

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--limit_scenes", type=int, default=0)
    ap.add_argument("--no_lpips", action="store_true")
    args = ap.parse_args()
    out = score(args.dataset, args.model, args.device,
                limit_scenes=args.limit_scenes, with_lpips=not args.no_lpips)
    agg = {k: round(v["mean"], 4) for k, v in out["aggregates"].items()}
    print(f"[eval] {args.model} on {args.dataset}: n={out['meta']['n_samples']} {agg}")


if __name__ == "__main__":
    main()
