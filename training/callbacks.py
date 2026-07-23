"""Atomic checkpoint helpers."""

from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(state: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def save_light_checkpoint(model_state: dict, cfg: dict, path: Path) -> None:
    save_checkpoint({"model": model_state, "config": cfg}, path)
