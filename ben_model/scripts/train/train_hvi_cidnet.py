"""Train HVI-CIDNet (CVPR'25) from scratch on MIT-MI via its native trainer.

HVI-CIDNet trains in its HVI colour space with a multi-term loss (L1 + SSIM +
edge + perceptual, in both RGB and HVI), integral to the method, so we drive
the repo's own ``train.py`` (subprocess) using its ``lol_v1`` data path, which
expects ``<dir>/low`` + ``<dir>/high`` folders — exactly the ``hvi`` layout
materialized by build_pairs.py. One GPU pinned; from-scratch (no pretrained).

Budget: epochs default to a ~150k-iter equivalent on whatever pairs were
materialized (``--target_iters``), matching the BasicSR trio instead of the
paper's 400 epochs (which on MIT-MI's ~22k pairs would be ~1.1M iters). Resume
warm-starts from the latest ``weights/train/epoch_N.pth`` via ``--start_epoch``
(loads weights, then trains the epoch budget again — see caveat). The native
per-epoch loss/PSNR logs are harvested into the uniform reporting schema.
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

MODEL = "hvi_cidnet"

# The trainer prints "... lr={}." (format string adds a trailing period); the lr
# group must stop at the last digit so float() doesn't choke on "0.0001.".
_RE_LOSS = re.compile(r"Epoch\[(\d+)\]:\s*Loss:\s*([\d.eE+-]+)\s*\|\|\s*"
                      r"Learning rate:\s*lr=([\d.eE+-]*\d)")
_RE_VALROW = re.compile(r"^\|\s*(\d+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|")


def latest_epoch(repo: Path) -> int:
    eps = []
    for p in (repo / "weights" / "train").glob("epoch_*.pth"):
        m = re.search(r"epoch_(\d+)\.pth", p.name)
        if m:
            eps.append(int(m.group(1)))
    return max(eps, default=0)


def harvest(lines: list[str], repo: Path, out_dir: Path) -> dict:
    """loss_curve.csv from stdout + val_curve.csv from the native metrics*.md,
    both mirrored into a uniform train_log.json."""
    import csv
    import json
    loss = [(int(m.group(1)), float(m.group(2)), float(m.group(3))) for ln in lines
            if (m := _RE_LOSS.search(ln))]
    with (out_dir / "loss_curve.csv").open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["epoch", "loss", "lr"]); w.writerows(loss)

    # newest metrics<...>.md table -> val_curve (| Epochs | PSNR | SSIM | LPIPS |)
    mds = sorted((repo / "results" / "training").glob("metrics*.md"),
                 key=lambda p: p.stat().st_mtime)
    val = []
    if mds:
        shutil.copy(mds[-1], out_dir / "native_metrics.md")
        for ln in mds[-1].read_text().splitlines():
            if (m := _RE_VALROW.match(ln)):
                val.append((int(m.group(1)), float(m.group(2)),
                            float(m.group(3)), float(m.group(4))))
        if val:
            with (out_dir / "val_curve.csv").open("w", newline="") as f:
                w = csv.writer(f); w.writerow(["epoch", "psnr", "ssim", "lpips"]); w.writerows(val)
    (out_dir / "train_log.json").write_text(json.dumps({"loss": loss, "val": val}, indent=2))
    best = max(val, key=lambda v: v[1], default=None)  # v = (epoch, psnr, ssim, lpips)
    return {"loss_points": len(loss), "val_points": len(val),
            "final_loss": loss[-1][1] if loss else None,
            "best_val_psnr": best[1] if best else None,
            "best_val_epoch": best[0] if best else None}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train HVI-CIDNet on MIT-MI (from scratch)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=0,
                    help="0 => derived from --target_iters; explicit overrides")
    ap.add_argument("--target_iters", type=int, default=300000,
                    help="full-run budget (epochs derived to match); equal across all 5 "
                         "baselines and sized so each model trains < 48 h.")
    ap.add_argument("--batch", type=int, default=0, help="0 => 8")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=8,
                    help="dataloader workers (native default 16; 8 is plenty and "
                         "keeps host RAM modest). Training is I/O-bound for this 8 MB model.")
    ap.add_argument("--val_interval", type=int, default=10,
                    help="run the (expensive full-res) in-training validation every N epochs "
                         "(+ final). Checkpoints stay per-epoch. Headline scoring is eval_all.py.")
    ap.add_argument("--val_max_images", type=int, default=200,
                    help="cap in-training val to N images (deterministic stride over all "
                         "scenes). Full-res LPIPS on all 2450 is ~5 h/val. 0 = all. eval_all.py "
                         "still scores the full test sets.")
    ap.add_argument("--resume", action="store_true",
                    help="warm-start from latest weights/train/epoch_N.pth (--start_epoch N)")
    ap.add_argument("--data_root", default=str(config.DATA_PREP_OUT / "hvi"))
    ap.add_argument("--out_dir", default="")
    _common.add_wandb_args(ap)
    args = ap.parse_args()

    batch = args.batch or 8
    droot = Path(args.data_root).resolve()
    n_pairs = _common.count_pairs(droot / "train" / "low" / "*.png")
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = _common.epochs_for_iters(args.target_iters, batch, n_pairs)
    out_dir = Path(args.out_dir) if args.out_dir else config.full_weights_dir(MODEL)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo = config.repo(MODEL)
    # HVI train.py writes ./results/training/metrics*.md and ./weights/train/*.pth
    (repo / "results" / "training").mkdir(parents=True, exist_ok=True)
    (repo / "weights" / "train").mkdir(parents=True, exist_ok=True)

    start_epoch = latest_epoch(repo) if args.resume else 0

    cmd = [config.PYTHON, "train.py",
           "--dataset", "lol_v1",
           "--data_train_lol_v1", str(droot / "train"),
           "--data_val_lol_v1", str(droot / "eval" / "low"),
           "--data_valgt_lol_v1", str(droot / "eval" / "high") + "/",
           "--nEpochs", str(epochs), "--snapshots", "1",
           "--val_interval", str(args.val_interval),
           "--val_max_images", str(args.val_max_images),
           "--batchSize", str(batch), "--cropSize", str(args.crop),
           "--lr", str(args.lr), "--threads", str(args.threads),
           "--cos_restart", "true", "--cos_restart_cyclic", "false",
           "--start_warmup", "false", "--grad_clip", "true",
           "--start_epoch", str(start_epoch)]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["WANDB_MODE"] = "disabled"
    env["HF_HUB_OFFLINE"] = "1"
    print(f"[train_hvi_cidnet] epochs={epochs} batch={batch} "
          f"n_pairs={n_pairs} (~{epochs * n_pairs // max(batch,1)} iters) "
          f"resume={args.resume} start_epoch={start_epoch} -> {out_dir}")

    # wandb: our wrapper owns the run (the subprocess's own wandb stays disabled
    # via env above). Train loss/lr is parsed live from stdout; val metrics live
    # in the native metrics*.md and are logged post-harvest below.
    wb = _common.make_wandb(args, MODEL, {
        "model": MODEL, "trainer": "native_hvi_cidnet", "epochs": epochs,
        "target_iters": args.target_iters, "batch": batch, "crop": args.crop,
        "lr": args.lr, "n_train_pairs": n_pairs}, out_dir)
    steps_per_epoch = max(1, n_pairs // max(batch, 1))

    def on_line(ln: str) -> None:
        if (m := _RE_LOSS.search(ln)):
            wb.log_train({"loss": float(m.group(2)), "lr": float(m.group(3))},
                         int(m.group(1)) * steps_per_epoch)

    t0 = time.time()
    # tee + capture: training + checkpointing happen before HVI's internal val,
    # so we validate success by the produced checkpoint, not the exit code.
    rc, lines = _common.stream(cmd, cwd=str(repo), env=env, on_line=on_line)
    if rc != 0:
        print(f"[train_hvi_cidnet] note: trainer exited {rc}; checking for checkpoint anyway")

    log = harvest(lines, repo, out_dir)
    # Checkpoint selection follows HVI-CIDNet's original convention: the BEST epoch by
    # val PSNR (the repo ships best_PSNR.pth), not the last. Fall back to the last epoch
    # if no val table was parsed or that epoch's file is missing.
    ckdir = repo / "weights" / "train"
    produced = None
    best_ep = log.get("best_val_epoch")
    if best_ep is not None and (ckdir / f"epoch_{best_ep}.pth").exists():
        produced = ckdir / f"epoch_{best_ep}.pth"
        print(f"[train_hvi_cidnet] selecting best-val epoch {best_ep} "
              f"(PSNR={log.get('best_val_psnr')})")
    if produced is None:
        produced = ckdir / f"epoch_{epochs + start_epoch}.pth"
        if not produced.exists():
            cand = sorted(ckdir.glob("epoch_*.pth"), key=lambda p: p.stat().st_mtime)
            produced = cand[-1] if cand else None
        if produced is not None:
            print(f"[train_hvi_cidnet] best-val epoch unavailable; using {produced.name}")
    if produced is None or not produced.exists():
        raise RuntimeError("no HVI-CIDNet checkpoint produced under weights/train/")
    shutil.copy(produced, out_dir / "model.pth")
    # val metrics came from the native metrics*.md (not stdout) — log them now.
    import json as _json
    for v in _json.loads((out_dir / "train_log.json").read_text()).get("val", []):
        ep, psnr, ssim, lpips = v
        wb.log_val({"psnr": psnr, "ssim": ssim, "lpips": lpips}, ep * steps_per_epoch)
    _common.write_meta(out_dir, {
        "model": MODEL, "trainer": "native_hvi_cidnet",
        "epochs": epochs, "target_iters": args.target_iters,
        "approx_iters": epochs * n_pairs // max(batch, 1),
        "batch": batch, "crop": args.crop, "lr": args.lr,
        "n_train_pairs": n_pairs, "resume": args.resume, "start_epoch": start_epoch,
        "git_commit": _common.git_commit(), "device": f"cuda:{args.gpu}",
        "runtime_sec_this_run": time.time() - t0,
        "checkpoint": str(out_dir / "model.pth"),
        "loss_curve": str(out_dir / "loss_curve.csv"),
        "val_curve": str(out_dir / "val_curve.csv") if log["val_points"] else None,
        "wandb_run_id": wb.run_id,
        **log,
    })
    wb.finish()
    print(f"[train_hvi_cidnet] checkpoint -> {out_dir / 'model.pth'} | {log}")


if __name__ == "__main__":
    main()
