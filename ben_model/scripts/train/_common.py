"""Shared CLI + environment helpers for the training adapters."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ben_model/scripts
from lib import config  # noqa: E402
from lib import wandblog  # noqa: E402


def add_wandb_args(ap: argparse.ArgumentParser) -> None:
    """Shared wandb flags. Logging is ON by default (matches the dainet ablation
    runs, which use mode=online); ``--no_wandb`` makes it a no-op."""
    ap.add_argument("--no_wandb", dest="wandb", action="store_false", default=True,
                    help="disable wandb logging (default: log online, like the ablation runs)")
    ap.add_argument("--wandb_project", default=wandblog.DEFAULT_PROJECT,
                    help=f"wandb project (default {wandblog.DEFAULT_PROJECT}; WANDB_PROJECT env also honoured)")


def make_wandb(args, model: str, config_dict: dict, out_dir: Path) -> wandblog.BenchWandb:
    """Open a benchmark wandb run for ``model`` (resuming the prior run when the
    sidecar id is present and we are resuming)."""
    resume_id = wandblog.read_run_id(out_dir) if getattr(args, "resume", None) else None
    return wandblog.BenchWandb(
        model=model, config=config_dict, run_dir=out_dir,
        mode="online" if getattr(args, "wandb", True) else "disabled",
        project=getattr(args, "wandb_project", None), resume_id=resume_id,
    )


def base_parser(model: str, default_patch: int) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=f"Train {model} on MIT-MI (from scratch)")
    ap.add_argument("--gpu", type=int, default=0, help="physical GPU index to pin (1 GPU/job)")
    ap.add_argument("--iters", type=int, default=0,
                    help="0 => 300000 (the equal budget; sized so each model trains < 48 h)")
    ap.add_argument("--batch", type=int, default=0, help="0 => 4")
    ap.add_argument("--patch", type=int, default=default_patch)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--data_root", default=str(config.DATA_PREP_OUT / "basicsr"))
    ap.add_argument("--out_dir", default="")
    ap.add_argument("--num_workers", type=int, default=8,
                    help="DataLoader workers; 8 keeps the patch-256 models (RLN2) "
                         "fed. Lower on small-CPU boxes.")
    ap.add_argument("--resume", nargs="?", const="auto", default="",
                    help="'auto' (or path) -> continue from <out_dir>/ckpt_last.pth")
    ap.add_argument("--ckpt_every", type=int, default=2000,
                    help="iters between ckpt_last.pth saves (resume points)")
    ap.add_argument("--val_every", type=int, default=5000,
                    help="iters between probe-masked val passes (0=off) -> val_curve.csv "
                         "(PSNR/SSIM) so generalization is visible during training")
    ap.add_argument("--val_max_side", type=int, default=768,
                    help="cap val image long side (OOM-safe full-res val for ConvNeXt-XL)")
    add_wandb_args(ap)
    return ap


def resolve(args, model: str) -> dict:
    """Pin GPU, fill defaults, resolve output dir."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    iters = args.iters or 300000
    batch = args.batch or 4
    out_dir = Path(args.out_dir) if args.out_dir else config.full_weights_dir(model)
    return {"device": "cuda:0", "iters": iters, "batch": batch,
            "patch": args.patch, "lr": args.lr, "out_dir": out_dir,
            "data_root": Path(args.data_root), "num_workers": args.num_workers,
            "resume": args.resume, "ckpt_every": args.ckpt_every,
            "val_every": args.val_every, "val_max_side": args.val_max_side}


# ---------------------------------------------------------------------------
# Helpers shared by the native (subprocess) trainers — IFBlend / HVI-CIDNet.
# ---------------------------------------------------------------------------
def epochs_for_iters(target_iters: int, batch: int, n_train_pairs: int) -> int:
    """Epochs that approximate ``target_iters`` optimisation steps (>=1).

    Keeps the native trainers on the same ~150k-iter budget as the BasicSR trio,
    adapting to whatever pair count was materialized (full or partial --scenes)."""
    if n_train_pairs <= 0:
        return 1
    return max(1, math.ceil(target_iters * batch / n_train_pairs))


def count_pairs(*globs: Path) -> int:
    """Total files matched across one or more ``dir/pattern`` glob specs."""
    n = 0
    for g in globs:
        n += sum(1 for _ in Path(g.parent).glob(g.name))
    return n


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def stream(cmd: list[str], cwd: str, env: dict, on_line=None) -> tuple[int, list[str]]:
    """Run a subprocess, tee its stdout live to our console, and return
    ``(returncode, captured_lines)`` so the wrapper can harvest train/val logs.

    ``on_line(line)`` is invoked per output line (stripped) when given, so a
    native trainer can parse + log each epoch to wandb live."""
    # stderr is left attached to our terminal so a subprocess tqdm bar (tqdm
    # writes to stderr) renders live; stdout stays piped for log parsing.
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                            stderr=None, text=True, bufsize=1)
    lines: list[str] = []
    for line in proc.stdout:
        sys.stdout.write(line); sys.stdout.flush()
        stripped = line.rstrip("\n")
        lines.append(stripped)
        if on_line is not None:
            try:
                on_line(stripped)
            except Exception:
                pass
    proc.wait()
    return proc.returncode, lines


def write_meta(out_dir: Path, meta: dict) -> None:
    (Path(out_dir) / "train_meta.json").write_text(json.dumps(meta, indent=2))
