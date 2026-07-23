"""IFBlend inference on a benchmark dataset (native model)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config  # noqa: E402
from methods import _unified, _runner  # noqa: E402

MODEL = "ifblend"


def _build_fn(ckpt: Path):
    repo = config.repo("ifblend")

    def build(device: str):
        os.chdir(repo)                       # backbone is loaded relative to cwd
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from utils_model import get_model    # noqa: E402  (repo-local)
        net = get_model("ifblend").to(device).eval()
        state = torch.load(str(ckpt), map_location=device)
        sd = state.get("model_state_dict", state.get("model", state))
        sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
        net.load_state_dict(sd)
        return lambda t: net(t)
    return build


def main() -> None:
    args = _unified.base_parser(MODEL).parse_args()
    device = _unified.setup_gpu(args.gpu)
    ckpt = _unified.resolve_ckpt(MODEL, args.ckpt).resolve()
    # fp32 inference: IFBlend's ConvNeXt-XL GCB + dynamic convs overflow fp16 autocast
    # at full resolution on this Turing box (no bf16) -> inf/NaN -> black output. The
    # native model trains/validates in fp32; keep inference fp32 to match.
    _runner.run_inference(args.dataset, MODEL, _build_fn(ckpt), device,
                          limit_scenes=args.limit_scenes, max_side=args.max_side,
                          pad_mult=32, half=False)


if __name__ == "__main__":
    main()
