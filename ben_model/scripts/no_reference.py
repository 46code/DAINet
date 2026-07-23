"""No-reference quality metrics (NIQE, BRISQUE) via `pyiqa`.

Both metrics return NaN with a one-time warning when the dependency is
missing, so the benchmark still emits all other metrics in that case.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import torch

_PYIQA = None
_PYIQA_WARNED = False
_NIQE_MODEL = None
_BRISQUE_MODEL = None


def _ensure_pyiqa():
    global _PYIQA, _PYIQA_WARNED
    if _PYIQA is not None:
        return _PYIQA
    try:
        import pyiqa  # type: ignore

        _PYIQA = pyiqa
        return _PYIQA
    except Exception as exc:  # pragma: no cover - depends on optional dep
        if not _PYIQA_WARNED:
            warnings.warn(
                f"pyiqa not available ({exc}); NIQE/BRISQUE will return NaN. "
                "Install with `pip install pyiqa` to enable.",
                stacklevel=2,
            )
            _PYIQA_WARNED = True
        return None


def _to_4d_tensor(img) -> Optional[torch.Tensor]:
    if isinstance(img, np.ndarray):
        t = torch.from_numpy(img.astype(np.float32))
        if t.dim() == 3 and t.shape[-1] == 3:
            t = t.permute(2, 0, 1)
    else:
        t = img.float()
    if t.dim() == 3:
        t = t.unsqueeze(0)
    return t.clamp(0.0, 1.0)


def niqe_score(img) -> float:
    global _NIQE_MODEL
    pyiqa = _ensure_pyiqa()
    if pyiqa is None:
        return float("nan")
    if _NIQE_MODEL is None:
        _NIQE_MODEL = pyiqa.create_metric("niqe", as_loss=False)
    t = _to_4d_tensor(img)
    with torch.no_grad():
        return float(_NIQE_MODEL(t).item())


def brisque_score(img) -> float:
    global _BRISQUE_MODEL
    pyiqa = _ensure_pyiqa()
    if pyiqa is None:
        return float("nan")
    if _BRISQUE_MODEL is None:
        _BRISQUE_MODEL = pyiqa.create_metric("brisque", as_loss=False)
    t = _to_4d_tensor(img)
    with torch.no_grad():
        return float(_BRISQUE_MODEL(t).item())
