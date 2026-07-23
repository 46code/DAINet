"""dainet (ours) inference on a benchmark dataset.

Reuses DAINet's deployment path (``scripts/_infer_common.infer_single``):
RGB-only input, inline SAM2 + DSINE normals, learned direction head. No training
(uses the existing ``runs/dainet_full`` checkpoint). The SAM2 builder pins to
logical cuda:0, so we pin the physical GPU via CUDA_VISIBLE_DEVICES and pass
device='cuda:0' (per the project's documented constraint).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config, datasets  # noqa: E402

MODEL = "dainet"
DEFAULT_CKPT = config.BRACKETTOUCH / "runs" / "dainet_full" / "checkpoints" / "model_best.pt"
DEFAULT_SAM = "~/my_model/sam2_weights/sam2.1_hiera_large.pt"


def _save_png(arr01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(arr01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def main() -> None:
    ap = argparse.ArgumentParser(description="Run dainet inference on a benchmark dataset")
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--sam_weights", default=DEFAULT_SAM)
    ap.add_argument("--limit_scenes", type=int, default=0)
    ap.add_argument("--estimate_illuminant", action="store_true")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)   # SAM2 pins to logical cuda:0
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # avoid timm/HF hub hangs
    device = "cuda:0"

    sys.path.insert(0, str(config.BRACKETTOUCH / "scripts"))
    from _infer_common import load_model_from_checkpoint, infer_single  # noqa: E402

    model, _cfg = load_model_from_checkpoint(args.ckpt, device)
    preds_dir = config.RESULTS / args.dataset / MODEL / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    for s in datasets.iter_samples(args.dataset, limit_scenes=args.limit_scenes):
        bgr = cv2.imread(str(s.input_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb_u8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pred = infer_single(model, rgb_u8, net_hw=config.NET_HW, device=device,
                            sam_weights=args.sam_weights,
                            estimate_illuminant=args.estimate_illuminant)
        _save_png(pred, preds_dir / f"{s.key}.png")
        n += 1
    print(f"[run] dainet on {args.dataset}: {n} preds -> {preds_dir}")


if __name__ == "__main__":
    main()
