"""RLN2 (CC36) inference on a benchmark dataset.

CC36 loads the generic ImageNet ConvNeXt-XL backbone from ``./weights/`` at
construction, so we chdir into the repo (where the backbone is symlinked).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config  # noqa: E402
from methods import _unified, _runner  # noqa: E402

MODEL = "rln2"


def main() -> None:
    ap = _unified.base_parser(MODEL)
    ap.add_argument("--num_blocks", type=int, default=4)
    args = ap.parse_args()
    device = _unified.setup_gpu(args.gpu)
    ckpt = _unified.resolve_ckpt(MODEL, args.ckpt).resolve()
    repo = config.repo("rln2")
    (repo / "weights").mkdir(exist_ok=True)
    link = repo / "weights" / "convnext_xlarge_22k_1k_384_ema.pth"
    if not link.exists():
        link.symlink_to((config.BACKBONES / "convnext_xlarge_22k_1k_384_ema.pth").resolve())
    build = _unified.make_build_fn(
        repo, "basicsr/models/archs/cc36_arch.py", "CC36",
        dict(num_feats=16, num_blocks=args.num_blocks), ckpt, chdir_repo=True)
    _runner.run_inference(args.dataset, MODEL, build, device,
                          limit_scenes=args.limit_scenes, max_side=args.max_side, pad_mult=32)


if __name__ == "__main__":
    main()
