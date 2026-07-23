"""One-shot: scan every materials_mip2.png and write a stable taxonomy.

The MIT-MI dataset's materials maps use sparse integer class ids drawn
from a 36-class material vocabulary, but the specific ids that appear
vary scene-to-scene. To keep model code dataset-agnostic we build a
fixed mapping from "raw id on disk" → "contiguous training class id" and
save it next to the dataset at
`data/raw/mit_mi/material_taxonomy.json`. Frozen across splits so
train/val/test all see the same K.

Usage:

    python scripts/build_material_taxonomy.py \\
        --root data/raw/mit_mi/jpg \\
        --splits train \\
        --out data/raw/mit_mi/material_taxonomy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def collect_raw_ids(root: Path, splits: list[str]) -> set[int]:
    """Walk <root>/<split>/<scene>/materials_mip2.png and union the unique ids."""
    raws: set[int] = set()
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            print(f"[material_taxonomy] skip missing split: {split_dir}", file=sys.stderr)
            continue
        scenes = sorted(p for p in split_dir.iterdir() if p.is_dir())
        missing = 0
        for scene in tqdm(scenes, desc=f"[taxonomy] {split}", unit="scene", dynamic_ncols=True):
            p = scene / "materials_mip2.png"
            if not p.exists():
                missing += 1
                continue
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                missing += 1
                continue
            if img.ndim == 3:
                img = img[..., 0]
            for v in np.unique(img).tolist():
                raws.add(int(v))
        if missing:
            print(
                f"[material_taxonomy] {split}: {missing}/{len(scenes)} scenes "
                "had no materials_mip2.png (soft-skipped)",
                file=sys.stderr,
            )
    return raws


def build_taxonomy(raws: set[int]) -> dict:
    """Stable ascending order; emit a raw_to_contiguous dict + K."""
    ordered = sorted(raws)
    return {
        "raw_to_contiguous": {str(r): i for i, r in enumerate(ordered)},
        "K": len(ordered),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="root of jpg/ with train/, test/ inside")
    parser.add_argument("--splits", default="train")
    parser.add_argument(
        "--out",
        default="data/raw/mit_mi/material_taxonomy.json",
        help="output JSON path; default sits next to the dataset under data/raw/mit_mi/",
    )
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    raws = collect_raw_ids(Path(args.root), splits)
    if not raws:
        print(
            f"[material_taxonomy] no materials_mip2.png under {args.root}/{{{','.join(splits)}}}",
            file=sys.stderr,
        )
        return 2
    taxonomy = build_taxonomy(raws)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(taxonomy, indent=2, sort_keys=True))
    print(
        f"[material_taxonomy] wrote {out_path} with K={taxonomy['K']} "
        f"(raw ids: {sorted(raws)})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
