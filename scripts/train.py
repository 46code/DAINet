"""dainet training entry point.

Usage:
    python scripts/train.py --config configs/dainet.yaml
    python scripts/train.py --config configs/dainet.yaml --dry_run
    python scripts/train.py --config configs/dainet.yaml --resume checkpoints/latest.pt

`--dry_run` instantiates dataset/model/loss/trainer and prints the parameter
count without launching training. Useful for sanity-checking a config without
burning GPU time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Quick pre-parse: if the user passed `--gpu N` on the command line, set
# `CUDA_VISIBLE_DEVICES` BEFORE importing torch so third-party libs don't
# silently initialise a CUDA context on the default physical GPU 0.
_early_gpu_index = None
for i, a in enumerate(sys.argv):
    if a == "--gpu" and i + 1 < len(sys.argv):
        try:
            _early_gpu_index = int(sys.argv[i + 1])
        except Exception:
            _early_gpu_index = None
        break
if _early_gpu_index is not None:
    # Export as string; this masks all other GPUs so the process only sees
    # the requested device. The logical device inside the process becomes
    # `cuda:0`.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_early_gpu_index))

# Disable HuggingFace hub network access to prevent model download hangs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_HOME", os.path.expanduser("~/.cache/timm"))

import argparse
import torch
import yaml

# Make the project importable when invoked as `python scripts/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Inputs are fixed 512x640; benchmark mode lets cuDNN pick the fastest
# conv algorithm for these shapes once and reuse it for the whole run.
torch.backends.cudnn.benchmark = True
# TF32 matmul/conv — free ~1.1-1.3x on Ampere/Turing, no capacity or quality
# change under the bf16 training path (loss math stays fp32).
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from training.trainer import Trainer  # noqa: E402


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume: a checkpoint path, or 'auto' for <paths.checkpoint_dir>/latest.pt. "
        "Restores model/optimizer/EMA/epoch/step and continues the same wandb run.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Build everything, print summary, exit.")
    parser.add_argument("--max_steps", type=int, default=None, help="Stop training after this many optimizer steps (for smoke runs).")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override `training.epochs` from the config.")
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    # A dry run only builds the trainer to sanity-check the config — it must
    # NOT open a (real, online) wandb run, which would leave a dangling /
    # crashed entry in the project since fit() never runs.
    if args.dry_run:
        cfg.setdefault("wandb", {})["mode"] = "disabled"
    # If we masked visible GPUs above (`CUDA_VISIBLE_DEVICES`), the
    # requested physical GPU becomes logical `cuda:0` inside the process.
    if args.gpu is not None and torch.cuda.is_available():
        if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
            device = "cuda:0"
        else:
            device = f"cuda:{args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Resolve resume BEFORE building the trainer so the same wandb run id can
    # be passed to wandb.init (continues one curve instead of orphaning it).
    resume_path = None
    if args.resume:
        ckpt_dir = Path(cfg.get("paths", {}).get("checkpoint_dir", "checkpoints"))
        resume_path = (ckpt_dir / "latest.pt") if args.resume == "auto" else Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"[dainet train] --resume target not found: {resume_path}")
        if not args.dry_run:
            sidecar = Path(cfg.get("paths", {}).get("log_dir", "logs")) / "wandb_id.txt"
            if sidecar.exists() and sidecar.read_text().strip():
                cfg.setdefault("wandb", {})["id"] = sidecar.read_text().strip()

    trainer = Trainer(cfg=cfg, device=device)

    start_epoch, start_step, batch_offset = 0, 0, 0
    if resume_path is not None and not args.dry_run:
        start_epoch, start_step, batch_offset = trainer.load_resume(resume_path)

    if args.dry_run:
        print("[dainet train] dry run complete")
        return

    summary = trainer.fit(
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        start_epoch=start_epoch,
        start_step=start_step,
        batch_offset=batch_offset,
        resume=resume_path is not None,
    )
    print(
        f"[dainet train] done best={summary['best_val']:.4f} "
        f"epoch={summary['best_epoch']} steps={summary['steps']:,}"
    )


if __name__ == "__main__":
    main()
