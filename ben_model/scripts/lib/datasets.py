"""Unified loaders for the 4 benchmark evaluation datasets.

Each dataset is exposed through the same ``Sample`` interface so the per-model
inference adapters and the scorer can treat them identically:

  - mit_mi    : 30 scenes x 25 directions, GT = target_clean, probe-masked.
  - ambient6k : 500 paired images  in/<id>_in.png  <-> gt/<id>_gt.png
  - cl3an     : 85 scenes  GT/<id>_GT.png  <-> IN_SH/<id>/<id>_1_IN.png
  - wsrd24    : 100 paired images  in/<name>.png  <-> gt/<name>.png

A ``Sample`` carries only paths + identifiers; pixel data is read on demand
via :func:`read_rgb01` and the probe mask via :func:`load_mask`. The ``key``
field is the stable prediction filename stem (mit_mi keys are
``"<scene>/dir_<id>"`` to preserve the per-scene/per-direction structure —
mit_mi has 25 directions per scene; the other datasets use ``"<id>"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from . import config
from . import probe_mask as pm


@dataclass
class Sample:
    dataset: str
    scene: str
    direction_id: int
    key: str
    input_path: Path
    gt_path: Path
    meta_path: Optional[Path] = None
    category: Optional[str] = None


def read_rgb01(path: Path) -> Optional[np.ndarray]:
    """Read an image as float32 RGB in [0, 1]. Returns None on failure."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_mask(sample: Sample, hw: tuple[int, int]) -> np.ndarray:
    """Probe mask (1=valid) for a sample at resolution ``hw``."""
    if sample.dataset == "mit_mi" and sample.meta_path is not None:
        return pm.mitmi_probe_mask(sample.meta_path, hw)
    return pm.full_mask(hw)


# ---------------------------------------------------------------------------
# Per-dataset scene listing + sample iteration
# ---------------------------------------------------------------------------

def _mitmi_scenes() -> list[str]:
    return sorted(p.name for p in config.MITMI_TEST_IN.iterdir() if p.is_dir())


def _iter_mitmi(scenes: list[str]) -> Iterator[Sample]:
    for scene in scenes:
        gt = config.MITMI_TEST_GT / scene / "target_clean.jpg"
        meta = config.MITMI_TEST_IN / scene / "meta.json"
        if not gt.exists():
            continue
        for did in range(config.N_DIRECTIONS):
            ip = config.MITMI_TEST_IN / scene / f"dir_{did}_mip2.jpg"
            if not ip.exists():
                continue
            yield Sample("mit_mi", scene, did, f"{scene}/dir_{did}", ip, gt, meta)


def _ambient6k_scenes() -> list[str]:
    return sorted(p.name[:-len("_in.png")] for p in (config.AMBIENT6K / "in").glob("*_in.png"))


def _iter_ambient6k(scenes: list[str]) -> Iterator[Sample]:
    for sid in scenes:
        ip = config.AMBIENT6K / "in" / f"{sid}_in.png"
        gt = config.AMBIENT6K / "gt" / f"{sid}_gt.png"
        if ip.exists() and gt.exists():
            yield Sample("ambient6k", sid, 0, sid, ip, gt)


def _cl3an_scenes() -> list[str]:
    return sorted((f.name[:-len("_GT.png")] for f in (config.CL3AN / "GT").glob("*_GT.png")),
                  key=lambda s: (len(s), s))


def _iter_cl3an(scenes: list[str]) -> Iterator[Sample]:
    for sid in scenes:
        gt = config.CL3AN / "GT" / f"{sid}_GT.png"
        ip = config.CL3AN / "IN_SH" / sid / f"{sid}_1_IN.png"
        if ip.exists() and gt.exists():
            yield Sample("cl3an", sid, 0, sid, ip, gt)


def _wsrd24_scenes() -> list[str]:
    return sorted(p.stem for p in (config.WSRD24 / "in").glob("*.png"))


def _iter_wsrd24(scenes: list[str]) -> Iterator[Sample]:
    for sid in scenes:
        ip = config.WSRD24 / "in" / f"{sid}.png"
        gt = config.WSRD24 / "gt" / f"{sid}.png"
        if ip.exists() and gt.exists():
            yield Sample("wsrd24", sid, 0, sid, ip, gt)


_SCENE_FN = {
    "mit_mi": _mitmi_scenes,
    "ambient6k": _ambient6k_scenes,
    "cl3an": _cl3an_scenes,
    "wsrd24": _wsrd24_scenes,
}
_ITER_FN = {
    "mit_mi": _iter_mitmi,
    "ambient6k": _iter_ambient6k,
    "cl3an": _iter_cl3an,
    "wsrd24": _iter_wsrd24,
}


def list_scenes(dataset: str, limit_scenes: int = 0) -> list[str]:
    if dataset not in _SCENE_FN:
        raise ValueError(f"unknown dataset {dataset!r}; valid: {list(_SCENE_FN)}")
    scenes = _SCENE_FN[dataset]()
    if limit_scenes and limit_scenes > 0:
        scenes = scenes[:limit_scenes]
    return scenes


def iter_samples(dataset: str, scenes: Optional[list[str]] = None,
                 limit_scenes: int = 0) -> Iterator[Sample]:
    if scenes is None:
        scenes = list_scenes(dataset, limit_scenes=limit_scenes)
    elif limit_scenes and limit_scenes > 0:
        scenes = scenes[:limit_scenes]
    yield from _ITER_FN[dataset](scenes)
