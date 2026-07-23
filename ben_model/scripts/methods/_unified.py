"""Loader for the unified-L1-trained BasicSR baselines (Restormer/Retinexformer/RLN2).

Each was trained from its repo architecture; the checkpoint is
``weights/<model>_mitmi/model.pth`` = {"model": state_dict}.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config  # noqa: E402
from train._simple_trainer import load_arch_class  # noqa: E402


def resolve_ckpt(model: str, override: str = "") -> Path:
    if override:
        return Path(override)
    return config.full_weights_dir(model) / "model.pth"


def make_build_fn(repo: Path, module: str, classname: str, ctor_kwargs: dict,
                  ckpt: Path, chdir_repo: bool = False) -> Callable:
    def build(device: str):
        if chdir_repo:
            os.chdir(repo)
        Arch = load_arch_class(repo, module, classname)
        net = Arch(**ctor_kwargs).to(device).eval()
        state = torch.load(str(ckpt), map_location=device)
        sd = state.get("model", state)
        net.load_state_dict(sd)
        return lambda t: net(t)
    return build


def base_parser(model: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Run {model} inference on a benchmark dataset")
    ap.add_argument("--dataset", required=True, choices=list(config.DATASETS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--limit_scenes", type=int, default=0)
    ap.add_argument("--max_side", type=int, default=0, help="cap inference long side (0=native)")
    return ap


def setup_gpu(gpu: int) -> str:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    return "cuda:0" if torch.cuda.is_available() else "cpu"
