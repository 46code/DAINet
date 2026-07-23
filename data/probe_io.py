"""Load pre-cropped chrome / gray sphere probe images.

MIT-MI ships per-direction 256x256 JPGs at
`<scene>/probes/dir_<N>_chrome256.jpg` and `dir_<N>_gray256.jpg`. Returns the
image as an RGB float32 array in [0, 1], or None if the file is missing.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _load_jpg(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0


def load_chrome_probe(scene_dir: Path | str, direction_id: int) -> np.ndarray | None:
    return _load_jpg(Path(scene_dir) / "probes" / f"dir_{direction_id}_chrome256.jpg")


def load_gray_probe(scene_dir: Path | str, direction_id: int) -> np.ndarray | None:
    return _load_jpg(Path(scene_dir) / "probes" / f"dir_{direction_id}_gray256.jpg")
