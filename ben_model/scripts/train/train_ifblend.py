"""Train IFBlend (ECCV'24) from scratch on MIT-MI via its native trainer.

IFBlend uses a custom multi-term loss (L1 + SSIM + gradient + perceptual) that
is integral to the method, so we drive the repo's own ``train.py`` (subprocess)
rather than the unified L1 trainer. Data is materialized into IFBlend's ImageSet
layout (``ifblend_ASRD6K/{Train,Validation}/<key>_d<id>_{in,gt}.png``). The
ConvNeXt-XL ImageNet backbone (generic, allowed) is symlinked into the repo.
wandb is disabled; one GPU is pinned.

Budget: epochs default to a ~150k-iter equivalent on whatever pairs were
materialized (``--target_iters``), so IFBlend trains the same amount as the
BasicSR trio instead of its paper's 200 epochs (which on MIT-MI's ~22k pairs
would be ~2.2M iters / weeks). Resume warm-starts from the native *best*
checkpoint (``--load 1``); see the caveat below. Train/val curves printed by the
native trainer are harvested into the uniform reporting schema.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib import config       # noqa: E402
from train import _common    # noqa: E402

MODEL = "ifblend"

_RE_TRAIN = re.compile(r"Epoch:\s*(\d+)\s*-\s*training_loss\s*([\d.eE+-]+)")
_RE_VAL = re.compile(r"Epoch:\s*(\d+)\s*-\s*validation MSE:\s*([\d.eE+-]+)\s*-\s*"
                     r"PSNR:\s*([\d.eE+-]+)\s*-\s*SSIM:\s*([\d.eE+-]+)")


def harvest(lines: list[str], out_dir: Path) -> dict:
    """Parse the native trainer's stdout into loss_curve.csv / val_curve.csv /
    train_log.json (uniform with the unified trainer's reporting)."""
    import csv
    import json
    train = [(int(m.group(1)), float(m.group(2))) for ln in lines
             if (m := _RE_TRAIN.search(ln))]
    val = [(int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4)))
           for ln in lines if (m := _RE_VAL.search(ln))]
    with (out_dir / "loss_curve.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "train_loss"]); w.writerows(train)
    if val:
        with (out_dir / "val_curve.csv").open("w", newline="") as f:
            w = csv.writer(f); w.writerow(["epoch", "mse", "psnr", "ssim"]); w.writerows(val)
    (out_dir / "train_log.json").write_text(json.dumps(
        {"train_loss": train, "val": val}, indent=2))
    return {"train_points": len(train), "val_points": len(val),
            "final_train_loss": train[-1][1] if train else None,
            "best_val_psnr": max((v[2] for v in val), default=None)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train IFBlend on MIT-MI (from scratch)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 => ~14 (150k-iter equiv); explicit overrides")
    ap.add_argument("--target_iters", type=int, default=300000,
                    help="full-run budget (epochs derived to match); equal across all 5 "
                         "baselines and sized so each model trains < 48 h.")
    ap.add_argument("--batch", type=int, default=0, help="0 => 2")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--resume", action="store_true",
                    help="warm-start from native best ckpt (--load 1); epoch counter restarts")
    ap.add_argument("--data_src", default=str(config.DATA_PREP_OUT / "ifblend_ASRD6K"))
    ap.add_argument("--out_dir", default="")
    _common.add_wandb_args(ap)
    args = ap.parse_args()

    batch = args.batch or 2
    data_src = Path(args.data_src).resolve()
    n_pairs = _common.count_pairs(data_src / "Train" / "*_in.png")
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = _common.epochs_for_iters(args.target_iters, batch, n_pairs)
    # Proportional LR decay mirroring IFBlend's original recipe (decay at 75% of
    # training, n_steps=2, gamma=0.6 -> final LR ~= 0.36x). The native MultiStepLR
    # uses step_sz=(epochs-decay_epoch)//n_steps; the old decay_epoch=epochs-1
    # collapsed step_sz to 0 (milestones=[E-1,E-1]) -> a constant LR, no annealing.
    # Guard step_sz>=1 so the decay actually fires.
    n_steps = 2
    decay_epoch = min(max(1, round(epochs * 0.75)), max(1, epochs - n_steps))
    out_dir = Path(args.out_dir) if args.out_dir else config.full_weights_dir(MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = config.repo(MODEL)
    # generic ImageNet ConvNeXt-XL backbone at ./weights/ (loaded by the model)
    (repo / "weights").mkdir(exist_ok=True)
    link = repo / "weights" / "convnext_xlarge_22k_1k_384_ema.pth"
    if not link.exists():
        link.symlink_to((config.BACKBONES / "convnext_xlarge_22k_1k_384_ema.pth").resolve())

    desc = f"{MODEL}_mitmi"
    work = out_dir / "work"
    cmd = [config.PYTHON, "train.py",
           "--model_name", "ifblend",
           "--data_src", str(data_src),
           "--n_epochs", str(epochs), "--decay_epoch", str(decay_epoch),
           "--n_steps", str(n_steps),
           "--batch_size", str(batch), "--lr", str(args.lr),
           "--img_height", str(args.size), "--img_width", str(args.size),
           "--save_checkpoint", "1", "--valid_checkpoint", "1",
           "--description", desc,
           "--ckp_dir", str((work / "ckpt").resolve()),
           "--res_dir", str((work / "res").resolve()),
           "--n_cpu", "4"]
    if args.resume:
        # native warm-start from the best checkpoint of this experiment.
        # NOTE: IFBlend's loop restarts the epoch counter from 0, so this resumes
        # *weights*, then re-runs the epoch budget (not a step-exact resume).
        cmd += ["--load", "1", "--load_from", desc]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    env["HF_HUB_OFFLINE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"  # flush native trainer stdout live (per-iter progress)
    print(f"[train_ifblend] epochs={epochs} batch={batch} "
          f"n_pairs={n_pairs} (~{epochs * n_pairs // max(batch,1)} iters) "
          f"decay_epoch={decay_epoch} n_steps={n_steps} "
          f"(milestones={[decay_epoch + k * ((epochs - decay_epoch) // n_steps) for k in range(n_steps)]}, "
          f"gamma=0.6) resume={args.resume} -> {out_dir}")

    # wandb: our wrapper owns the run (the subprocess's own wandb stays disabled
    # via env above); we log each native epoch live by parsing its stdout.
    wb = _common.make_wandb(args, MODEL, {
        "model": MODEL, "trainer": "native_ifblend", "epochs": epochs,
        "target_iters": args.target_iters, "batch": batch, "size": args.size,
        "lr": args.lr, "decay_epoch": decay_epoch, "n_steps": n_steps,
        "n_train_pairs": n_pairs}, out_dir)
    steps_per_epoch = max(1, n_pairs // max(batch, 1))

    def on_line(ln: str) -> None:
        if (m := _RE_TRAIN.search(ln)):
            wb.log_train({"loss": float(m.group(2))}, int(m.group(1)) * steps_per_epoch)
        elif (m := _RE_VAL.search(ln)):
            ep = int(m.group(1))
            wb.log_val({"mse": float(m.group(2)), "psnr": float(m.group(3)),
                        "ssim": float(m.group(4))}, ep * steps_per_epoch)

    t0 = time.time()
    rc, lines = _common.stream(cmd, cwd=str(repo), env=env, on_line=on_line)
    if rc != 0:
        print(f"[train_ifblend] note: trainer exited {rc}; checking for checkpoint anyway")

    # locate produced checkpoint (best/, else latest epoch) and copy out.
    ckroot = work / "ckpt" / desc
    cand = ckroot / "best" / "checkpoint.pt"
    if not cand.exists():
        epdirs = sorted((p for p in ckroot.glob("*/checkpoint.pt")),
                        key=lambda p: p.stat().st_mtime)
        cand = epdirs[-1] if epdirs else None
    if cand is None or not cand.exists():
        raise RuntimeError(f"no IFBlend checkpoint produced under {ckroot}")
    shutil.copy(cand, out_dir / "model.pth")

    log = harvest(lines, out_dir)
    _common.write_meta(out_dir, {
        "model": MODEL, "trainer": "native_ifblend",
        "epochs": epochs, "target_iters": args.target_iters,
        "approx_iters": epochs * n_pairs // max(batch, 1),
        "decay_epoch": decay_epoch, "n_steps": n_steps, "lr_gamma": 0.6,
        "batch": batch, "size": args.size, "lr": args.lr,
        "n_train_pairs": n_pairs, "resume": args.resume,
        "git_commit": _common.git_commit(), "device": f"cuda:{args.gpu}",
        "runtime_sec_this_run": time.time() - t0,
        "checkpoint": str(out_dir / "model.pth"),
        "loss_curve": str(out_dir / "loss_curve.csv"),
        "val_curve": str(out_dir / "val_curve.csv") if log["val_points"] else None,
        "wandb_run_id": wb.run_id,
        **log,
    })
    wb.finish()
    print(f"[train_ifblend] checkpoint -> {out_dir / 'model.pth'} | {log}")


if __name__ == "__main__":
    main()
