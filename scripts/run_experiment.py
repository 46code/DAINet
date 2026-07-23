"""dainet experiment launcher — one CLI for baseline + every ablation.

Each named experiment (see ``experiments/registry.py``) is launched with all
outputs isolated under ``runs/<name>/``:

    runs/<name>/
      ├── config.yaml        # fully-resolved snapshot (reproducible)
      ├── checkpoints/       # latest.pt, model_best.pt, model_final.pt
      ├── logs/              # iter_history.jsonl, epoch_summary.json,
      │                      #   metrics_history.csv, wandb/
      ├── plots/             # val_samples/, val_failures/
      └── report/            # written later by scripts/make_report.py

Usage:
    python scripts/run_experiment.py --list
    python scripts/run_experiment.py --exp dainet_full --gpu 0
    python scripts/run_experiment.py --exp abl_no_xdir --gpu 0
    python scripts/run_experiment.py --exp abl_no_xdir --gpu 0 --resume auto   # continue
    python scripts/run_experiment.py --exp dainet_full --max_steps 30 --dry_run

The fully-resolved config is written BEFORE training starts, so a run is
reproducible from ``runs/<name>/config.yaml`` even via plain
``scripts/train.py --config runs/<name>/config.yaml``.
"""

from __future__ import annotations

import os
import sys

# Disable HuggingFace hub network access to prevent model download hangs.
# Pre-trained weights must be cached locally or we fall back to random init.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_HOME", os.path.expanduser("~/.cache/timm"))

from pathlib import Path

# Pre-parse --gpu BEFORE importing torch so CUDA_VISIBLE_DEVICES masks
# other devices (mirrors scripts/train.py).
_early_gpu_index = None
for _i, _a in enumerate(sys.argv):
    if _a == "--gpu" and _i + 1 < len(sys.argv):
        try:
            _early_gpu_index = int(sys.argv[_i + 1])
        except Exception:
            _early_gpu_index = None
        break
if _early_gpu_index is not None:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(_early_gpu_index))

import argparse
from copy import deepcopy

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.registry import (  # noqa: E402
    PRIORITY_ORDER,
    experiments_by_priority,
    get_experiment,
    validate_registry,
)


def deep_merge(base: dict, patch: dict) -> dict:
    """Recursively merge ``patch`` onto a deep copy of ``base``."""
    out = deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def _resolve_resume(resume_arg: str | None, run_dir: Path) -> Path | None:
    """Map ``--resume`` to a checkpoint path. ``auto`` → the run's latest.pt."""
    if not resume_arg:
        return None
    p = (run_dir / "checkpoints" / "latest.pt") if resume_arg == "auto" else Path(resume_arg)
    if not p.exists():
        raise SystemExit(f"[run_experiment] --resume target not found: {p}")
    return p


def _peek_wandb_id(run_dir: Path) -> str | None:
    """Cheap wandb-run-id recovery for resume: prefer the sidecar
    ``logs/wandb_id.txt``, else parse it from a ``logs/wandb/run-*-<id>/`` dir
    (so even a checkpoint that predates wandb-id tracking resumes one curve)."""
    sidecar = run_dir / "logs" / "wandb_id.txt"
    if sidecar.exists():
        rid = sidecar.read_text().strip()
        if rid:
            return rid
    wb = run_dir / "logs" / "wandb"
    if wb.exists():
        cands = sorted(wb.glob("run-*-*"))
        if cands:
            return cands[-1].name.rsplit("-", 1)[-1]
    return None


def resolve_config(base_path: str, exp_name: str, runs_root: str) -> tuple[dict, Path]:
    """Build the resolved config for an experiment and route its paths."""
    with open(base_path) as f:
        base = yaml.safe_load(f)
    spec = get_experiment(exp_name)
    cfg = deep_merge(base, spec.get("overrides", {}))

    run_dir = Path(runs_root) / exp_name
    paths = cfg.setdefault("paths", {})
    paths["checkpoint_dir"] = str(run_dir / "checkpoints")
    paths["log_dir"] = str(run_dir / "logs")
    paths["plot_dir"] = str(run_dir / "plots")

    wb = cfg.setdefault("wandb", {})
    wb["run_name"] = exp_name

    # Record provenance so the snapshot is self-describing.
    cfg["_experiment"] = {
        "name": exp_name,
        "description": spec.get("description", ""),
        "base_config": str(base_path),
    }
    return cfg, run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default=None, help="Experiment name (see --list).")
    parser.add_argument("--base", default="configs/dainet.yaml", help="Base config.")
    parser.add_argument("--runs_root", default="runs", help="Root dir for per-run output.")
    parser.add_argument("--list", action="store_true", help="List experiments (grouped by priority) and exit.")
    parser.add_argument(
        "--priority",
        default=None,
        choices=list(PRIORITY_ORDER),
        help="With --list, show only this priority tier (high|moderate|low).",
    )
    parser.add_argument("--dry_run", action="store_true", help="Resolve + snapshot config, build trainer, no fit (no wandb run).")
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume training: a checkpoint path, or 'auto' for runs/<exp>/checkpoints/latest.pt. "
        "Restores model/optimizer/EMA/epoch/step and continues the same wandb run.",
    )
    parser.add_argument("--gpu", type=int, default=None)
    args = parser.parse_args()

    # One-knob guard: fail fast if a single-component ablation drifted to >1 knob.
    validate_registry(args.base)

    if args.list or not args.exp:
        rows = experiments_by_priority(args.priority)
        print("Available experiments (run in this order — high → moderate → low):\n")
        cur_tier = None
        for name, tier, desc in rows:
            if tier != cur_tier:
                cur_tier = tier
                print(f"  [{tier.upper()}]")
            print(f"    {name:<24s} {desc}")
        if not args.exp:
            return
        print()

    cfg, run_dir = resolve_config(args.base, args.exp, args.runs_root)
    # A dry run only builds dataset/model/loss/trainer to sanity-check the
    # config — it must NOT open a (real, online) wandb run, which would leave
    # a dangling/crashed entry in the project since fit() never runs.
    if args.dry_run:
        cfg.setdefault("wandb", {})["mode"] = "disabled"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.yaml"
    snapshot.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"[run_experiment] exp={args.exp}  run_dir={run_dir}")
    print(f"[run_experiment] resolved config snapshot → {snapshot}")

    # Resolve resume BEFORE building the trainer so the same wandb run is
    # continued (the id has to be passed to wandb.init).
    resume_path = _resolve_resume(args.resume, run_dir)
    if resume_path is not None and not args.dry_run:
        rid = _peek_wandb_id(run_dir)
        if rid:
            cfg.setdefault("wandb", {})["id"] = rid
            print(f"[run_experiment] resuming wandb run id={rid}")

    # Import torch + Trainer only now (after CUDA_VISIBLE_DEVICES is set).
    import torch

    torch.backends.cudnn.benchmark = True
    # TF32 matmul/conv — free ~1.1-1.3x on Ampere/Turing, no capacity or
    # quality change under the bf16 training path (loss math stays fp32).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from training.trainer import Trainer

    if args.gpu is not None and torch.cuda.is_available():
        device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is not None else f"cuda:{args.gpu}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = Trainer(cfg=cfg, device=device)

    start_epoch, start_step, batch_offset = 0, 0, 0
    if resume_path is not None and not args.dry_run:
        start_epoch, start_step, batch_offset = trainer.load_resume(resume_path)

    if args.dry_run:
        print("[run_experiment] dry run complete")
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
        f"[run_experiment] done exp={args.exp} best={summary['best_val']:.4f} "
        f"epoch={summary['best_epoch']} steps={summary['steps']:,}"
    )


if __name__ == "__main__":
    main()
