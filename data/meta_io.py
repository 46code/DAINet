"""Parse `meta.json` for a MIT-MI scene.

The exact key layout in MIT-MI's `meta.json` varies across distributions. This
parser is defensive: it looks for `directions` as either a list of dicts or a
dict keyed by direction id, extracts `phi`, `theta`, and
`brightness_normalization` per direction, and returns a uniform structure. If
keys are missing, sensible defaults (`phi=0`, `theta=0`,
`brightness_normalization=1`) are used and a `present=False` flag is set.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _coerce_dir_entry(entry: dict[str, Any], did: int) -> dict[str, Any]:
    return {
        "phi": float(entry.get("phi", 0.0)),
        "theta": float(entry.get("theta", 0.0)),
        "brightness_normalization": float(entry.get("brightness_normalization", 1.0)),
        "chrome_bbox": entry.get("chrome_bbox") or entry.get("chrome", {}).get("bbox") if isinstance(entry.get("chrome"), dict) else None,
        "gray_bbox": entry.get("gray_bbox") or entry.get("gray", {}).get("bbox") if isinstance(entry.get("gray"), dict) else None,
        "direction_id": did,
        "present": True,
    }


def load_meta(scene_dir: Path | str) -> dict[str, Any]:
    """Load and parse meta.json under scene_dir.

    Returns:
        {
            "directions": {direction_id (int): {phi, theta, brightness_normalization, ...}, ...},
            "present": bool,
        }
        If meta.json is missing, directions is {} and present is False.
    """
    scene_dir = Path(scene_dir)
    p = scene_dir / "meta.json"
    if not p.exists():
        return {"directions": {}, "present": False}
    return _load_meta_cached(str(p))


@lru_cache(maxsize=512)
def _load_meta_cached(path_str: str) -> dict[str, Any]:
    with open(path_str) as f:
        raw = json.load(f)
    dirs = raw.get("directions")
    out: dict[int, dict[str, Any]] = {}
    if isinstance(dirs, list):
        for d in dirs:
            did = int(d.get("direction_id", -1))
            if did < 0:
                continue
            out[did] = _coerce_dir_entry(d, did)
    elif isinstance(dirs, dict):
        for k, v in dirs.items():
            try:
                did = int(k)
            except (TypeError, ValueError):
                continue
            out[did] = _coerce_dir_entry(v, did)
    return {"directions": out, "present": True}


def default_dir_entry(did: int) -> dict[str, Any]:
    """Fallback used when a direction has no meta.json entry."""
    return {
        "phi": 0.0,
        "theta": 0.0,
        "brightness_normalization": 1.0,
        "chrome_bbox": None,
        "gray_bbox": None,
        "direction_id": did,
        "present": False,
    }
