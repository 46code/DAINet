"""Restormer inference on a benchmark dataset."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config  # noqa: E402
from methods import _unified, _runner  # noqa: E402

MODEL = "restormer"


def main() -> None:
    args = _unified.base_parser(MODEL).parse_args()
    device = _unified.setup_gpu(args.gpu)
    ckpt = _unified.resolve_ckpt(MODEL, args.ckpt)
    build = _unified.make_build_fn(
        config.repo("restormer"), "basicsr/models/archs/restormer_arch.py", "Restormer",
        dict(inp_channels=3, out_channels=3, dim=48, num_blocks=[4, 6, 6, 8],
             num_refinement_blocks=4, heads=[1, 2, 4, 8], ffn_expansion_factor=2.66,
             bias=False, LayerNorm_type="WithBias"), ckpt)
    _runner.run_inference(args.dataset, MODEL, build, device,
                          limit_scenes=args.limit_scenes, max_side=args.max_side, pad_mult=8)


if __name__ == "__main__":
    main()
