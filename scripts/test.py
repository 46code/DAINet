"""dainet metric scoring — train + validation + held-out test, probe-masked.

Loads a checkpoint and scores the three benchmark metrics (PSNR, MS-SSIM,
LPIPS) on each requested split with the chrome+gray calibration probes masked
out. The model is run at FULL CAPACITY from a single
sRGB input on every split: it is given only the image plus the priors computed
from it — SAM2 ids (cache) and DSINE normals — while illumination is supplied by
the learned illum_token + predicted (φ,θ,b) direction head (no GT metadata is
fed in). This matches the deployment path (scripts/infer.py), so the numbers are
directly comparable and the train↔val↔test gap is itself a generalisation result.

Splits:
  - ``test``  — the held-out MIT-MI test scenes (``paths.test_jpg_root`` /
                ``paths.test_gt_root``; doubly-nested layout). Hard-fails if the
                probes are not masked (benchmark-fairness requirement).
  - ``val``   — the internal held-out validation scenes (10% of ``train/``).
  - ``train`` — the training scenes, re-built with augmentation OFF (a train-fit
                number). Capped by ``--max_batches_train`` since it is large.

Writes ``<paths.log_dir>/metrics_by_split.json`` = ``{"train":…,"val":…,
"test":…}`` and keeps ``<paths.log_dir>/test_metrics.json`` (the test sub-dict)
for back-compat. Per-run reporting figures are produced offline by
``scripts/make_report.py --run runs/<name>``.

Usage:
    python scripts/test.py --config runs/baseline/config.yaml \\
        --ckpt runs/baseline/checkpoints/model_best.pt --gpu 0
    # only the held-out test split:
    python scripts/test.py --config runs/baseline/config.yaml \\
        --ckpt runs/baseline/checkpoints/model_best.pt --splits test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

# dainet policy: test never logs to wandb. Force-disable before any import that
# may transitively initialize a run.
os.environ["WANDB_MODE"] = "disabled"

# Disable HuggingFace hub network access to prevent model download hangs.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TIMM_HOME", os.path.expanduser("~/.cache/timm"))

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset import DAINetDataset  # noqa: E402
from data.splits import subset_scenes  # noqa: E402
from evaluation.metrics import MetricComputer, psnr, ms_ssim, lpips_score  # noqa: E402
from models.network import DAINet  # noqa: E402
from training.trainer import Trainer  # noqa: E402


def _enumerate_scene_dirs(root: Path) -> list[str]:
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def _build_test_model(cfg: dict, device: str) -> torch.nn.Module:
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    taxonomy_path = data_cfg.get(
        "material_taxonomy_path", "data/raw/mit_mi/material_taxonomy.json"
    )
    k_material = 0
    if data_cfg.get("use_material", True) and model_cfg.get("use_material_head", False):
        try:
            from data.material_io import num_classes as _material_num_classes

            k_material = int(_material_num_classes(taxonomy_path))
        except FileNotFoundError:
            k_material = 0

    model = DAINet(
        embed_dim=model_cfg.get("embed_dim", 128),
        seg_embed_dim=model_cfg.get("seg_embed_dim", 128),
        illum_hidden=model_cfg.get("illum_hidden", 128),
        attn_heads=model_cfg.get("attn_heads", 8),
        sh_l_max=model_cfg.get("sh_l_max", 4),
        pretrained_encoder=model_cfg.get("pretrained_encoder", True),
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        use_swin_bottleneck=model_cfg.get("use_swin_bottleneck", False),
        use_specular_head=model_cfg.get("use_specular_head", False),
        use_illum_token=model_cfg.get("use_illum_token", False),
        use_reflectance_memory=model_cfg.get("use_reflectance_memory", False),
        reflectance_memory_size=model_cfg.get("reflectance_memory_size", 256),
        use_material_head=bool(model_cfg.get("use_material_head", False)) and k_material > 0,
        material_num_classes=k_material,
        material_head_hidden=int(model_cfg.get("material_head_hidden", 256)),
        use_normals=bool(model_cfg.get("use_normals", False)),
        normals_fusion=model_cfg.get("normals_fusion", None),
        use_sam_conditioning=bool(model_cfg.get("use_sam_conditioning", True)),
        direction_encoding=str(model_cfg.get("direction_encoding", "continuous")),
        num_directions=int(model_cfg.get("num_directions", 25)),
        use_illum_chroma_field=bool(model_cfg.get("use_illum_chroma_field", False)),
        use_illuminant_head=bool(model_cfg.get("use_illuminant_head", False)),
        use_direction_head=bool(model_cfg.get("use_direction_head", False)),
        activation_checkpoint=bool(cfg.get("training", {}).get("activation_checkpoint", False)),
    ).to(device)

    channels_last = bool(model_cfg.get("channels_last", True)) and torch.cuda.is_available()
    if channels_last:
        model = model.to(memory_format=torch.channels_last)

    if bool(cfg.get("training", {}).get("compile", False)) and hasattr(torch, "compile"):
        model = torch.compile(model)

    return model


def _build_test_dataset(cfg: dict) -> DAINetDataset:
    paths = cfg.get("paths", {})
    ds_cfg = cfg.get("dataset", {})
    size = tuple(cfg.get("spatial", {}).get("size", [512, 640]))
    subset = float(ds_cfg.get("subset_ratio", 1.0))

    if paths.get("test_jpg_root") and paths.get("test_gt_root"):
        jpg_root = Path(paths["test_jpg_root"])
        jpg_gt_root = Path(paths["test_gt_root"])
    else:
        jpg_root = Path(paths.get("jpg_root", "data/raw/mit_mi/jpg")) / "test"
        jpg_gt_root = Path(paths.get("jpg_gt_root", "data/raw/mit_mi/jpg_gt")) / "test"

    pool = _enumerate_scene_dirs(jpg_root)
    pool = subset_scenes(pool, ratio=subset)

    return DAINetDataset(
        jpg_root,
        jpg_gt_root,
        pool,
        size=size,
        augment=False,
        mode="test",
        split="test",
        sam_root=paths.get("sam_root", "data/raw/mit_mi/sam_masks"),
        normals_root=paths.get("normals_root", "data/raw/mit_mi/normals"),
    )


def _score_split(
    model,
    dataset,
    *,
    device: str,
    batch_size: int,
    num_workers: int,
    max_batches: int | None,
    require_probe: bool,
    label: str,
    collect_per_image: bool = False,
) -> tuple[dict[str, float], list[dict] | None]:
    """Score one split RGB-only with probe masking; return the metrics dict.

    ``require_probe`` hard-fails when no probe pixels were masked (the
    benchmark-fairness invariant for the test split); for train/val a missing
    mask is only warned about. ``max_batches`` caps the pass (the train split
    is large) — when set, the loader is shuffled with a fixed seed so the
    subset spans scenes rather than the first few alphabetically.
    """
    shuffle = max_batches is not None
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(42) if shuffle else None,
    )
    metrics = MetricComputer(with_lpips=True)
    per_rows: list[dict] | None = [] if collect_per_image else None
    n_imgs = n_imgs_masked = masked_px = total_px = 0
    total_iters = min(max_batches, len(loader)) if max_batches is not None else len(loader)
    start_time = time.monotonic()
    last_heartbeat = start_time

    def _heartbeat(done_batches: int) -> None:
        nonlocal last_heartbeat
        now = time.monotonic()
        if done_batches < total_iters and now - last_heartbeat < 30:
            return
        elapsed = now - start_time
        rate = done_batches / elapsed if elapsed > 0 else 0.0
        remaining = max(total_iters - done_batches, 0)
        eta = remaining / rate if rate > 0 else float("inf")
        eta_text = f"{eta / 60:.1f}m" if eta != float("inf") else "?"
        print(
            f"[dainet test] {label}: heartbeat {done_batches}/{total_iters} batches "
            f"({elapsed / 60:.1f}m elapsed, ETA {eta_text})",
            file=sys.stderr,
            flush=True,
        )
        last_heartbeat = now

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, total=total_iters, desc=label, leave=False)):
            if max_batches is not None and i >= max_batches:
                break
            _heartbeat(i)
            rgb = batch["input_rgb"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            probe_mask = batch.get("probe_mask")
            if probe_mask is not None:
                probe_mask = probe_mask.to(device, non_blocking=True)
                zero_per_img = (probe_mask == 0).flatten(1).sum(dim=1)
                px_per_img = probe_mask[0].numel()
                n_imgs += probe_mask.shape[0]
                n_imgs_masked += int((zero_per_img > 0).sum().item())
                masked_px += int(zero_per_img.sum().item())
                total_px += int(px_per_img * probe_mask.shape[0])
            else:
                n_imgs += rgb.shape[0]
            # Single-RGB-input contract at FULL capacity: the model is given only
            # the sRGB image plus the priors computed from it (SAM2 ids + DSINE
            # normals). (φ,θ,b) are NOT passed — the direction head predicts them.
            seg = batch.get("input_seg")
            if seg is not None:
                seg = seg.to(device, non_blocking=True)
            normals = batch.get("normals")
            if normals is not None:
                normals = normals.to(device, non_blocking=True)
            out = model(rgb, seg=seg, normals=normals, compute_material=False)
            metrics.update(out["output"], target, segments=None, mask=probe_mask)
            if per_rows is not None:
                scenes = batch.get("scene", ["?"] * rgb.shape[0])
                dir_ids = batch.get("direction_id")
                psnr_b = psnr(out["output"].float().clamp(0, 1), target.float().clamp(0, 1), mask=probe_mask)
                msssim_b = ms_ssim(out["output"].float().clamp(0, 1), target.float().clamp(0, 1), mask=probe_mask)
                lpips_b = lpips_score(out["output"].float().clamp(0, 1), target.float().clamp(0, 1), mask=probe_mask)
                for j in range(rgb.shape[0]):
                    did = int(dir_ids[j].item()) if dir_ids is not None else j + len(per_rows)
                    key = f"{scenes[j]}_{did}"
                    per_rows.append({"key": key, "psnr": psnr_b[j].item(),
                                     "ms_ssim": msssim_b[j].item(), "lpips": lpips_b[j].item()})

    _heartbeat(total_iters)

    if n_imgs_masked == 0:
        msg = (
            f"[dainet test] {label}: probe masking took effect on 0/{n_imgs} "
            "images. Check each scene's meta.json (chrome/gray boundaries)."
        )
        if require_probe:
            raise SystemExit(
                msg + " Refusing to report test metrics that score the probes."
            )
        print("WARNING: " + msg)
    else:
        pct = 100.0 * masked_px / max(total_px, 1)
        print(f"[dainet test] {label}: probe-masked {n_imgs_masked}/{n_imgs} "
              f"images, mean {pct:.2f}% px masked")

    final = metrics.compute()
    final.pop("seg_delta_e_var", None)  # seg-aware key is not in the fair set
    print(f"[dainet test] {label}: " + "  ".join(f"{k}={v:.4f}" for k, v in final.items()))
    return final, per_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--splits", default="train,val,test",
        help="Comma-separated subset of {train,val,test} to score (default all).",
    )
    parser.add_argument(
        "--max_batches_train", type=int, default=150,
        help="Cap the (large) train-split pass at this many batches; <=0 = full.",
    )
    parser.add_argument(
        "--per_image", action="store_true",
        help="Also write per_image_test_metrics.json (required for ablation_significance.py).",
    )
    args = parser.parse_args()

    requested = [s.strip() for s in args.splits.split(",") if s.strip()]
    unknown = [s for s in requested if s not in ("train", "val", "test")]
    if unknown:
        raise SystemExit(f"[dainet test] unknown split(s): {unknown}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # dainet policy: test never logs to wandb. WANDB_MODE=disabled (set above) is
    # not enough — Trainer passes the config mode explicitly to wandb.init().
    cfg.setdefault("wandb", {})["mode"] = "disabled"

    device = (
        f"cuda:{args.gpu}"
        if args.gpu is not None and torch.cuda.is_available()
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    bs = args.batch_size or cfg.get("training", {}).get("batch_size", 6)
    num_workers = cfg.get("training", {}).get("num_workers", 4)
    cap_train = args.max_batches_train if args.max_batches_train > 0 else None

    log_dir = Path(cfg.get("paths", {}).get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    if requested == ["test"]:
        print(f"[dainet test] building lightweight test path on {device}", flush=True)
        model = _build_test_model(cfg, device)
        print(f"[dainet test] loading checkpoint {args.ckpt}", flush=True)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        state = ckpt["model"] if "model" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()

        ds = _build_test_dataset(cfg)
        print(f"[dainet test] scoring split=test ({len(ds):,} samples)", flush=True)
        agg, per_rows = _score_split(
            model,
            ds,
            device=device,
            batch_size=bs,
            num_workers=num_workers,
            max_batches=None,
            require_probe=True,
            label="test",
            collect_per_image=args.per_image,
        )

        by_split: dict[str, dict[str, float]] = {"test": agg}
        if per_rows is not None:
            p = log_dir / "per_image_test_metrics.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(per_rows, indent=2))
            print(f"[dainet test] wrote {p} ({len(per_rows)} rows)")

        (log_dir / "metrics_by_split.json").write_text(json.dumps(by_split, indent=2))
        print(f"[dainet test] wrote {log_dir / 'metrics_by_split.json'}")
        (log_dir / "test_metrics.json").write_text(json.dumps(by_split["test"], indent=2))
        print(f"[dainet test] wrote {log_dir / 'test_metrics.json'}")
        return

    trainer = Trainer(cfg=cfg, device=device)
    if getattr(trainer.wandb, "enabled", False) or getattr(trainer.wandb, "run", None) is not None:
        raise SystemExit("[dainet test] refusing to run: a wandb run was opened (set wandb.mode=disabled).")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    trainer.model.load_state_dict(state)
    trainer.model.eval()

    by_split: dict[str, dict[str, float]] = {}
    for split in requested:
        if split == "test":
            ds = trainer.build_test_dataset()
            if len(ds) == 0:
                jpg_root, jpg_gt_root = trainer._split_roots("test")
                raise SystemExit(
                    f"[dainet test] no test scenes found under {jpg_root} (gt: "
                    f"{jpg_gt_root}). Set paths.test_jpg_root / paths.test_gt_root."
                )
            cap, require = None, True
        elif split == "val":
            ds, cap, require = trainer.val_ds, None, False
        else:  # train
            ds = trainer.build_train_eval_dataset()
            if ds is None or len(ds) == 0:
                print(f"[dainet test] train: no train-eval dataset available — skipped.")
                continue
            cap, require = cap_train, False

        print(f"[dainet test] scoring split={split} ({len(ds):,} samples"
              + (f", capped at {cap} batches)" if cap else ")"))
        want_per_image = args.per_image and split == "test"
        agg, per_rows = _score_split(
            trainer.model, ds, device=device, batch_size=bs, num_workers=num_workers,
            max_batches=cap, require_probe=require, label=split,
            collect_per_image=want_per_image,
        )
        by_split[split] = agg
        if per_rows is not None:
            p = log_dir / "per_image_test_metrics.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(per_rows, indent=2))
            print(f"[dainet test] wrote {p} ({len(per_rows)} rows)")

    (log_dir / "metrics_by_split.json").write_text(json.dumps(by_split, indent=2))
    print(f"[dainet test] wrote {log_dir / 'metrics_by_split.json'}")
    if "test" in by_split:  # back-compat: keep the single test_metrics.json
        (log_dir / "test_metrics.json").write_text(json.dumps(by_split["test"], indent=2))
        print(f"[dainet test] wrote {log_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
