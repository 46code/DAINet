"""Unified L1 trainer for the BasicSR-family baselines (Restormer / Retinexformer / RLN2).

All three repos configure a *pure L1 pixel loss* (``pixel_opt: L1Loss``) for the
restoration objective, so a single shared trainer with L1 + Adam + cosine
schedule + random-crop/flip augmentation reproduces their training objective
faithfully while guaranteeing an identical, fair protocol across the three.
(The richer-loss custom baselines IFBlend / HVI-CIDNet keep their native
trainers instead.) Importing the per-model architecture is left to the caller;
this module only owns the data pipeline and the optimisation loop.

Resume & reporting
------------------
* ``ckpt_last.pth`` (model+opt+sched+scaler+step) is written every ``ckpt_every``
  iters and at the end; ``resume="auto"`` (or a path) restores it and continues
  to ``iters`` total. The eval/runners still read the plain ``model.pth``.
* For reporting, every run writes a uniform set of artifacts into ``out_dir``:
  ``loss_curve.csv`` (step,loss,lr,sec_per_it), an optional ``val_curve.csv``
  (probe-masked val on the materialized ``val`` split, via the project metric
  backbone) when ``val_every>0``, and a full config/provenance ``train_meta.json``.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_arch_class(repo: Path, module_relpath: str, class_name: str):
    """Import an architecture class from a repo file without importing basicsr."""
    repo = Path(repo)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))  # let any vendored intra-repo import resolve
    path = repo / module_relpath
    spec = importlib.util.spec_from_file_location(f"_arch_{class_name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)


class PairedCrops(Dataset):
    """Paired lq/gt folders (matching filenames) -> random crops in [0,1] CHW."""

    def __init__(self, root: Path, patch: int, augment: bool, length: int | None = None):
        self.lq_dir = Path(root) / "lq"
        self.gt_dir = Path(root) / "gt"
        self.files = sorted(p.name for p in self.lq_dir.glob("*.png"))
        if not self.files:
            raise FileNotFoundError(f"no lq pngs under {self.lq_dir}")
        self.patch = patch
        self.augment = augment
        self.length = length or len(self.files)

    def __len__(self):
        return self.length

    def _read(self, path: Path) -> np.ndarray:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, idx):
        name = self.files[idx % len(self.files)]
        lq = self._read(self.lq_dir / name)
        gt = self._read(self.gt_dir / name)
        h, w = lq.shape[:2]
        ph = min(self.patch, h, w)
        if h > ph:
            top = np.random.randint(0, h - ph + 1)
            left = np.random.randint(0, w - ph + 1)
            lq = lq[top:top + ph, left:left + ph]
            gt = gt[top:top + ph, left:left + ph]
        else:
            lq = cv2.resize(lq, (ph, ph)); gt = cv2.resize(gt, (ph, ph))
        if self.augment:
            if np.random.rand() < 0.5:
                lq, gt = lq[:, ::-1], gt[:, ::-1]
            if np.random.rand() < 0.5:
                lq, gt = lq[::-1], gt[::-1]
        lq = torch.from_numpy(np.ascontiguousarray(lq)).permute(2, 0, 1)
        gt = torch.from_numpy(np.ascontiguousarray(gt)).permute(2, 0, 1)
        return lq, gt


def _match_size(out: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    if out.shape[-2:] != gt.shape[-2:]:
        out = F.interpolate(out, size=gt.shape[-2:], mode="bilinear", align_corners=False)
    return out


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


@torch.no_grad()
def _validate(model: torch.nn.Module, val_root: Path, device: str,
              max_images: int = 60, max_side: int = 1024) -> dict:
    """Light probe-masked val on the materialized val split via the project metric
    backbone. Full-resolution, batch 1, capped at ``max_images`` for speed."""
    from lib import metricsx  # lazy: pulls evaluation.metrics + lpips

    lq_dir, gt_dir = val_root / "lq", val_root / "gt"
    files = sorted(p.name for p in lq_dir.glob("*.png"))[:max_images]
    if not files:
        return {}
    keys = ["l1", "psnr", "ssim"]
    acc = {k: [] for k in keys}
    was_training = model.training
    model.eval()
    for name in files:
        lq = cv2.cvtColor(cv2.imread(str(lq_dir / name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        gt = cv2.cvtColor(cv2.imread(str(gt_dir / name)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        h, w = lq.shape[:2]
        if max_side and max(h, w) > max_side:
            s = max_side / max(h, w)
            lq = cv2.resize(lq, (int(w * s), int(h * s)))
            gt = cv2.resize(gt, (int(w * s), int(h * s)))
        lt = torch.from_numpy(lq).permute(2, 0, 1)[None].to(device)
        gtt = torch.from_numpy(gt).permute(2, 0, 1)[None].to(device)
        # pad to /8 so fully-conv nets accept arbitrary sizes
        H, W = lt.shape[-2:]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        out = model(F.pad(lt, (0, pw, 0, ph), mode="reflect"))[..., :H, :W]
        out = _match_size(out, gtt).float().clamp(0, 1)
        acc["l1"].append(float((out - gtt).abs().mean().item()))
        row = metricsx.score_pair(out, gtt, mask=None, with_lpips=False)
        for k in ("psnr", "ssim"):
            acc[k].append(row[k])
    if was_training:
        model.train()
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def train_model(model: torch.nn.Module, data_root: Path, out_dir: Path, *,
                iters: int, batch: int, patch: int, lr: float, device: str,
                val_every: int = 0, log_every: int = 10, amp: bool = True,
                grad_clip: float = 0.01, num_workers: int = 4,
                resume: str = "", ckpt_every: int = 2000,
                model_name: str = "", arch_kwargs: dict | None = None,
                val_max_images: int = 60, val_max_side: int = 768,
                wandb_run=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(iters, 1), eta_min=lr * 1e-2)
    scaler = torch.amp.GradScaler("cuda", enabled=amp and "cuda" in device)
    l1 = torch.nn.L1Loss()

    # ---- resume -----------------------------------------------------------
    last_ckpt = out_dir / "ckpt_last.pth"
    resume_path = None
    if resume == "auto":
        resume_path = last_ckpt if last_ckpt.exists() else None
    elif resume:
        resume_path = Path(resume)
    start_step = 0
    if resume_path and resume_path.exists():
        ck = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"]); start_step = int(ck["step"])
        print(f"  [resume] {resume_path} @ step {start_step}/{iters}", flush=True)

    val_root = Path(data_root) / "val"
    fresh = start_step == 0
    loss_csv = out_dir / "loss_curve.csv"
    val_csv = out_dir / "val_curve.csv"
    lf = loss_csv.open("w" if fresh else "a", newline="")
    lw = csv.writer(lf)
    if fresh:
        lw.writerow(["step", "loss", "lr", "sec_per_it"])

    def save_last(step: int) -> None:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                    "step": step}, last_ckpt)

    # ---- train loop -------------------------------------------------------
    remaining = max(iters - start_step, 0)
    losses, it_times, t0, step = [], [], time.time(), start_step
    best_psnr, best_step = -float("inf"), None  # best-by-val checkpoint selection
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    if remaining > 0:
        ds = PairedCrops(Path(data_root) / "train", patch, augment=True,
                         length=max(remaining * batch, 1))
        # Throughput: pinned memory + prefetch keep the (heavy, full-PNG-decode)
        # workers ahead of the GPU so per-iter time stays steady instead of
        # spiking when the buffer drains (esp. at patch=256, e.g. RLN2).
        dl_kwargs = dict(batch_size=batch, shuffle=True, num_workers=num_workers,
                         drop_last=True, pin_memory="cuda" in device)
        if num_workers > 0:
            dl_kwargs.update(persistent_workers=True, prefetch_factor=4)
        dl = DataLoader(ds, **dl_kwargs)
        t_prev = time.time()
        pbar = tqdm(dl, total=remaining, desc=model_name or "train", dynamic_ncols=True)
        for lq, gt in pbar:
            if step >= iters:
                break
            lq = lq.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp and "cuda" in device):
                out = _match_size(model(lq), gt)
                loss = l1(out.float().clamp(0, 1), gt)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt); scaler.update(); sched.step()
            lv = float(loss.detach().item())
            if not np.isfinite(lv):
                raise RuntimeError(f"non-finite loss at iter {step}")
            step += 1
            now = time.time(); spi = now - t_prev; t_prev = now
            losses.append(lv); it_times.append(spi)
            if step % log_every == 0 or step == start_step + 1:
                cur_lr = sched.get_last_lr()[0]
                pbar.set_postfix(L1="{:.4f}".format(lv), lr="{:.2e}".format(cur_lr))
                lw.writerow([step, f"{lv:.6f}", f"{cur_lr:.3e}", f"{spi:.4f}"])
                lf.flush()
                print(f"  [iter {step}/{iters}] L1={lv:.4f} lr={cur_lr:.2e} "
                      f"({spi:.2f}s/it)", flush=True)
                if wandb_run is not None:
                    wandb_run.log_train({"loss": lv, "lr": cur_lr, "sec_per_it": spi}, step)
            if val_every and step % val_every == 0 and val_root.exists():
                vr = _validate(model, val_root, device, max_images=val_max_images,
                               max_side=val_max_side)
                if vr:
                    new = not val_csv.exists()
                    with val_csv.open("a", newline="") as vf:
                        vw = csv.writer(vf)
                        if new:
                            vw.writerow(["step", "l1", "psnr", "ssim"])
                        vw.writerow([step, f"{vr.get('l1', float('nan')):.6f}",
                                     f"{vr.get('psnr', float('nan')):.4f}",
                                     f"{vr.get('ssim', float('nan')):.4f}"])
                    print(f"  [val {step}] psnr={vr.get('psnr'):.3f} "
                          f"ssim={vr.get('ssim'):.4f}", flush=True)
                    if wandb_run is not None:
                        wandb_run.log_val(vr, step)
                    # Best-by-val checkpoint (matches the originals: IFBlend /
                    # Retinexformer / RLN2 / HVI all report best-val-PSNR).
                    p = vr.get("psnr")
                    if p is not None and p > best_psnr:
                        best_psnr, best_step = p, step
                        torch.save({"model": model.state_dict()}, out_dir / "model.pth")
                        print(f"  [val {step}] new best PSNR={p:.3f} -> model.pth",
                              flush=True)
            if ckpt_every and step % ckpt_every == 0:
                save_last(step)
    lf.close()

    # ---- final save + meta ------------------------------------------------
    save_last(step)  # ckpt_last.pth always = latest state (for step-exact resume)
    ckpt = out_dir / "model.pth"
    # model.pth = best-by-val checkpoint (saved during validation). Only fall back to
    # the last state if validation never ran (val_every=0) -> no best was recorded.
    if best_step is None:
        torch.save({"model": model.state_dict()}, ckpt)
    steady = float(np.median(it_times[2:])) if len(it_times) > 2 else (
        float(np.median(it_times)) if it_times else None)
    peak_vram_mb = (torch.cuda.max_memory_allocated() / 1e6
                    if device.startswith("cuda") else None)
    meta = {
        "model": model_name, "trainer": "unified_l1",
        "iters_done": step, "iters_requested": iters,
        "lr": lr, "batch": batch, "patch": patch, "amp": amp,
        "grad_clip": grad_clip, "num_workers": num_workers,
        "ckpt_every": ckpt_every, "resume": resume or None,
        "val_every": val_every, "arch_kwargs": arch_kwargs or {},
        "ckpt_selection": "best_val_psnr" if best_step is not None else "last",
        "best_val_psnr": best_psnr if best_step is not None else None,
        "best_val_step": best_step,
        "device": device, "data_root": str(data_root),
        "git_commit": _git_commit(),
        "final_loss": losses[-1] if losses else None,
        "mean_last10": float(np.mean(losses[-10:])) if losses else None,
        "sec_per_it_steady": steady,
        "peak_vram_mb": peak_vram_mb,
        "runtime_sec_this_run": time.time() - t0,
        "checkpoint": str(ckpt), "ckpt_last": str(last_ckpt),
        "loss_curve": str(loss_csv),
        "val_curve": str(val_csv) if val_csv.exists() else None,
        "wandb_run_id": getattr(wandb_run, "run_id", None),
    }
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2))
    if wandb_run is not None:
        wandb_run.finish()
    print(f"  [done] {step} iters, final L1={meta['final_loss']}, ckpt={ckpt}", flush=True)
    return meta
