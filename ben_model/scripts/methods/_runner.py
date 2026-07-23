"""Shared inference loop for the per-model runners.

A runner supplies ``build_fn(device) -> forward(tensor)`` where ``forward`` maps
a [1,3,H,W] float tensor in [0,1] to a [1,3,H,W] prediction in [0,1]. This
module handles dataset iteration, optional resolution capping (``max_side``),
padding to a multiple (so fully-convolutional nets accept arbitrary sizes),
half-precision autocast, and writing predictions to
``results/<dataset>/<model>/preds/<key>.png`` (mit_mi keeps the per-scene/dir
nesting — 25 directions per scene).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config, datasets  # noqa: E402


def _save_png(arr01: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor((np.clip(arr01, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3])


def _pad_to(t: torch.Tensor, mult: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, h, w = t.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    return t, (h, w)


def run_inference(dataset: str, model_name: str, build_fn: Callable, device: str,
                  *, limit_scenes: int = 0, max_side: int = 0, pad_mult: int = 32,
                  half: bool = True, results_root: Path | None = None) -> int:
    results_root = results_root or config.RESULTS
    preds_dir = results_root / dataset / model_name / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    forward = build_fn(device)
    use_amp = half and "cuda" in device
    n, t0 = 0, time.time()
    for s in datasets.iter_samples(dataset, limit_scenes=limit_scenes):
        rgb = datasets.read_rgb01(s.input_path)
        if rgb is None:
            continue
        H0, W0 = rgb.shape[:2]
        work = rgb
        if max_side and max(H0, W0) > max_side:
            scale = max_side / max(H0, W0)
            work = cv2.resize(rgb, (int(round(W0 * scale)), int(round(H0 * scale))),
                              interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(work).permute(2, 0, 1).unsqueeze(0).to(device)
        t, (h, w) = _pad_to(t, pad_mult)
        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = forward(t)
        out = out[..., :h, :w].float().clamp(0, 1)
        pred = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
        if pred.shape[:2] != (H0, W0):
            pred = cv2.resize(pred, (W0, H0), interpolation=cv2.INTER_LINEAR)
        _save_png(pred, preds_dir / f"{s.key}.png")
        n += 1
    dt = time.time() - t0
    print(f"[run] {model_name} on {dataset}: {n} preds in {dt:.1f}s -> {preds_dir}", flush=True)
    return n
