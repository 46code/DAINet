"""HVI-CIDNet inference on a benchmark dataset (native model)."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config  # noqa: E402
from methods import _unified, _runner  # noqa: E402

MODEL = "hvi_cidnet"


def _build_fn(ckpt: Path):
    repo = config.repo("hvi_cidnet")

    def build(device: str):
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from net.CIDNet import CIDNet  # noqa: E402  (repo-local)
        net = CIDNet().to(device).eval()
        state = torch.load(str(ckpt), map_location=device)
        sd = state.get("model", state)
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        return lambda t: net(t)
    return build


def main() -> None:
    args = _unified.base_parser(MODEL).parse_args()
    device = _unified.setup_gpu(args.gpu)
    ckpt = _unified.resolve_ckpt(MODEL, args.ckpt).resolve()
    # pad to a multiple of 16: CIDNet's U-Net skips mismatch on some odd sizes
    # at /8 (the 374-vs-375 case) when only padded to 8.
    _runner.run_inference(args.dataset, MODEL, _build_fn(ckpt), device,
                          limit_scenes=args.limit_scenes, max_side=args.max_side, pad_mult=16)


if __name__ == "__main__":
    main()
