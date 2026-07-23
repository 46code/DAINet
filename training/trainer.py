"""DAINet trainer.

End-to-end training loop with:
- AdamW + linear warmup + cosine annealing + AMP bf16 (no GradScaler)
- gradient accumulation + global-norm clipping
- per-iteration checkpointing (`training.checkpoint_interval_iters`)
- classifier-free conditioning dropout (`training.null_cond_prob`)
- step-based linear warmup envelope on the four "soft" loss terms
- interval validation (live + EMA) every `training.val_interval_iters`
- best-checkpoint selection by val PSNR

Val + test always call ``model(rgb)`` (RGB-only / null conditioning) so the
metric used for best-checkpoint selection matches the benchmark scenario
against jpg-only baselines. Training is the only place that supplies seg
+ illum, and even there it stochastically drops them so the null-token
fallback receives gradient signal.

No post-training plots — the durable artifacts are
`logs/iter_history.jsonl`, `logs/epoch_summary.json`, and
`logs/metrics_history.csv`. Paper figures are regenerated offline.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.amp
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import DAINetDataset, discover_scenes
from data.scene_pair_sampler import (
    ScenePairBatchSampler,
    SingleDirectionPerSceneSampler,
)
from data.splits import leave_k_directions_out, subset_scenes, train_val_split
from evaluation.failure_analysis import FailureBuckets
from evaluation.visualizer import (
    make_val_failure_pair_images,
    make_val_failures_grid,
    make_val_pair_images,
    make_val_samples_grid,
)
from losses.manager import DAINetLoss
from models.network import DAINet

from .callbacks import save_checkpoint
from .ema import EMAModel
from .evaluator import Evaluator
from .wandb_logger import WandbLogger


DEFAULT_JPG_ROOT = "data/raw/mit_mi/jpg"
DEFAULT_JPG_GT_ROOT = "data/raw/mit_mi/jpg_gt"


def _worker_init_fn(_worker_id: int) -> None:
    """Prevent OpenCV/OMP thread oversubscription inside DataLoader workers.

    With 8 workers each spawning its own multi-threaded BLAS / OpenCV
    pool, the CPU thrashes on context switches and data loading slows
    dramatically. Pin each worker to a single thread per native lib.
    """
    import os
    import cv2

    cv2.setNumThreads(0)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


class Trainer:
    def __init__(
        self,
        cfg: dict,
        train_ds: DAINetDataset | None = None,
        val_ds: DAINetDataset | None = None,
        device: str | None = None,
        wandb_logger: WandbLogger | None = None,
    ):
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve K_material from the committed taxonomy (if available). The
        # model + loss + dataset all need this — one source of truth, set
        # BEFORE _build_datasets so the dataset builder can read it.
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
        self._k_material = k_material

        # Datasets
        if train_ds is None or val_ds is None:
            train_ds, val_ds = self._build_datasets()
        self.train_ds = train_ds
        self.val_ds = val_ds

        # Model — ConvNeXt encoder (Tiny or Base) + SAM2 FiLM + null tokens.
        # ConvNeXt benefits substantially from channels-last on Ampere/Turing
        # under fp16 (~15-25% faster). Toggle off via model.channels_last=false
        # if a downstream module is incompatible.
        self.model = DAINet(
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
            activation_checkpoint=bool(
                cfg.get("training", {}).get("activation_checkpoint", False)
            ),
        ).to(self.device)
        self.channels_last = bool(model_cfg.get("channels_last", True)) and torch.cuda.is_available()
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)

        # Optional torch.compile (opt-in via training.compile). Wrap after the
        # model is on-device + in its final memory format. Off by default — see
        # configs/dainet.yaml. NOTE: compile adds an `_orig_mod.` prefix to
        # state_dict / param names, so a checkpoint must be resumed with the
        # SAME compile setting it was trained under (EMA + resume stay
        # self-consistent within one run either way).
        if bool(cfg.get("training", {}).get("compile", False)) and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
            print("[dainet] torch.compile enabled")

        # Loss — feed K_material through `options` so material_R_var knows
        # how many real classes to track without having to inspect logits.
        loss_options = dict(cfg.get("loss_options", {}))
        if k_material > 0:
            loss_options.setdefault("material_num_classes", k_material)
        self.loss_fn = DAINetLoss(
            weights=cfg.get("loss", {}),
            options=loss_options,
        ).to(self.device)

        # Training cfg
        train_cfg = cfg.get("training", {})
        self.batch_size = int(train_cfg.get("batch_size", 6))
        self.epochs = int(train_cfg.get("epochs", 25))
        self.grad_accum = int(train_cfg.get("grad_accum_steps", 1))
        self.grad_clip = float(train_cfg.get("grad_clip", 1.0))
        # AMP precision policy. bf16 is the default: it has the same 8-bit
        # exponent as fp32 (no Inf/subnormal at log/exp/div sites), so the
        # Retinex decomposition trains without GradScaler. fp16 kept as a
        # back-compat opt-in; "off" disables autocast entirely.
        amp_mode = str(train_cfg.get("amp", "bf16")).lower()
        if amp_mode not in ("bf16", "fp16", "off"):
            raise ValueError(f"training.amp must be one of bf16|fp16|off, got {amp_mode!r}")
        self.amp_enabled = amp_mode in ("bf16", "fp16") and torch.cuda.is_available()
        self.amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
        self.use_scaler = (self.amp_dtype == torch.float16) and self.amp_enabled
        self.log_interval = int(cfg.get("logging", {}).get("log_interval", 25))
        # Media (prediction / worst-failure image grids) to wandb. Default off
        # for speed — rendering them costs extra forward passes + uploads per
        # epoch. make_report.py regenerates qualitative figures offline.
        self.log_media = bool(cfg.get("logging", {}).get("log_media", True))
        self.ckpt_interval = int(train_cfg.get("checkpoint_interval_iters", 200))
        self.null_cond_prob = float(train_cfg.get("null_cond_prob", 0.2))
        # Direction-head teacher-forcing mix: on full-cond batches, feed the
        # head's PREDICTED (φ,θ,b) instead of GT with this probability so the
        # model adapts to the inference distribution (mitigates exposure bias).
        self.direction_pred_p = float(model_cfg.get("direction_pred_p", 0.5))
        self._use_direction_head = bool(getattr(self.model, "use_direction_head", False))
        self._cond_rng = random.Random(int(train_cfg.get("seed", 42)))
        # Provenance for retrain-free checkpoints / run_meta.json.
        self._git_sha = self._git_sha_now()
        self._best_selected_by: str | None = None

        # Interval validation cadence — full val + EMA every N optimizer
        # steps in addition to per-epoch. 0 = disabled (epoch-only).
        self.val_interval_iters = int(train_cfg.get("val_interval_iters", 0))

        opt_cfg = train_cfg.get("optimizer", {})
        self.base_lr = float(opt_cfg.get("lr", 3.0e-5))
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.base_lr,
            weight_decay=float(opt_cfg.get("weight_decay", 0.01)),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        sched_cfg = train_cfg.get("scheduler", {})
        self.warmup_epochs = int(sched_cfg.get("warmup_epochs", 3))
        self.min_lr = float(sched_cfg.get("min_lr", 5e-7))

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_scaler)

        # EMA — shadow weights for evaluation / checkpointing. Off by
        # default; enable via training.ema_decay > 0 in the YAML.
        self.ema_decay = float(train_cfg.get("ema_decay", 0.0))
        self.ema_eval = bool(train_cfg.get("ema_eval", False)) and self.ema_decay > 0
        # EMA val is the single most expensive thing in the epoch loop:
        # it doubles validation time. Default to every epoch (back-compat)
        # but allow gating it via training.ema_eval_every_n.
        self.ema_eval_every_n = int(train_cfg.get("ema_eval_every_n", 1))
        self.ema = EMAModel(self.model, decay=self.ema_decay) if self.ema_decay > 0 else None

        # Val subsampling — cap batches per epoch so val cost doesn't grow
        # linearly with val set size. Defaults to None = full val.
        self.val_max_batches = train_cfg.get("val_max_batches", None)
        if self.val_max_batches is not None:
            self.val_max_batches = int(self.val_max_batches)
        # Tighter cap applied to *interval* val passes only — epoch-end val
        # uses `val_max_batches` (typically the full pool). Keeps interval
        # val cheap (smooth iter-cadence curves) while preserving an honest
        # full-pool number for best-checkpoint selection at the epoch boundary.
        self.val_interval_max_batches = train_cfg.get("val_interval_max_batches", None)
        if self.val_interval_max_batches is not None:
            self.val_interval_max_batches = int(self.val_interval_max_batches)

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.es_enabled = bool(es_cfg.get("enabled", True))
        self.es_metric = es_cfg.get("metric", "psnr")
        self.es_mode = es_cfg.get("mode", "max")
        self.es_patience = int(es_cfg.get("patience", 6))
        self.es_min_delta = float(es_cfg.get("min_delta", 0.02))
        self._best_val = float("inf") if self.es_mode == "min" else -float("inf")
        self._best_epoch = -1
        self._stalls = 0

        # Paths — defensive .get() so a config missing `paths` still resolves
        # to the documented defaults (matches docs/cli.md).
        paths = cfg.get("paths", {})
        self.ckpt_dir = Path(paths.get("checkpoint_dir", "checkpoints"))
        self.log_dir = Path(paths.get("log_dir", "logs"))
        self.plot_dir = Path(paths.get("plot_dir", "plots"))

        # Structured JSON / CSV logging state. log_dir is created eagerly so
        # iter_history.jsonl is appendable from the first step.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._iter_jsonl_path = self.log_dir / "iter_history.jsonl"
        self._epoch_summary_path = self.log_dir / "epoch_summary.json"
        self._metrics_csv_path = self.log_dir / "metrics_history.csv"
        # Log-file lifecycle is decided at fit() start by _prepare_logs():
        # a fresh run truncates jsonl + rewrites the CSV header; a resume
        # appends to both and restores the in-memory epoch-summary list.
        self._metrics_csv_header_written = False
        self._epoch_summaries: list[dict] = []
        self._resume_wandb_id: str | None = None

        # Loaders — persistent_workers + prefetch_factor only kick in when
        # num_workers > 0, so guard the kwargs to keep num_workers=0 valid
        # (smoke tests use that path).
        nw = int(train_cfg.get("num_workers", 4))
        extra_loader_kwargs: dict = {}
        if nw > 0:
            extra_loader_kwargs["persistent_workers"] = bool(
                train_cfg.get("persistent_workers", False)
            )
            extra_loader_kwargs["prefetch_factor"] = int(
                train_cfg.get("prefetch_factor", 2)
            )
        worker_kwargs = {"worker_init_fn": _worker_init_fn} if nw > 0 else {}
        # One-direction-per-scene ablation (`abl_single_direction`): with a single
        # direction per scene there are no same-scene pairs, so ScenePairBatchSampler
        # (which requires ≥ 2 dirs/scene) would yield zero batches. Switch to a
        # plain one-dir-per-scene sampler instead; the cross-direction losses then
        # contribute zero by construction.
        seed = int(train_cfg.get("seed", 42))
        if int(self.cfg.get("dataset", {}).get("directions_per_scene") or 0) == 1:
            train_batch_sampler = SingleDirectionPerSceneSampler(
                train_ds.samples, batch_size=self.batch_size, seed=seed
            )
        else:
            train_batch_sampler = ScenePairBatchSampler(
                train_ds.samples, batch_size=self.batch_size
            )
        self.train_loader = DataLoader(
            train_ds,
            batch_sampler=train_batch_sampler,
            num_workers=nw,
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            **extra_loader_kwargs,
            **worker_kwargs,
        )
        # Val pass has no backward → use a larger batch and disable shuffle.
        val_bs = int(train_cfg.get("val_batch_size", max(self.batch_size * 2, 4)))
        self.val_loader = DataLoader(
            val_ds,
            batch_size=val_bs,
            shuffle=False,
            num_workers=nw,
            pin_memory=bool(train_cfg.get("pin_memory", True)),
            **extra_loader_kwargs,
            **worker_kwargs,
        )

        # Evaluator / logger
        self.evaluator = Evaluator(model=self.model, loss_fn=self.loss_fn, device=self.device)
        self.wandb = wandb_logger if wandb_logger is not None else WandbLogger(cfg, run_dir=self.log_dir)

        # Fixed val sample indices (so the same images are tracked across epochs)
        n_val_samples = int(cfg.get("logging", {}).get("val_sample_count", 4))
        n_val_samples = min(n_val_samples, len(val_ds))
        if n_val_samples == 0:
            self._val_sample_indices: list[int] = []
        elif n_val_samples == 1:
            self._val_sample_indices = [len(val_ds) // 2]
        else:
            stride = max(1, len(val_ds) // n_val_samples)
            self._val_sample_indices = [
                min(i * stride, len(val_ds) - 1) for i in range(n_val_samples)
            ]
        self._n_worst = int(cfg.get("logging", {}).get("worst_top_k", 3))

    # ---------------------------------------------------------------- paths
    def _split_roots(self, split: str) -> tuple[Path, Path]:
        """Resolve <jpg_root>/<split> and <jpg_gt_root>/<split>.

        Config-level paths point at the parent folder; the split lives one
        directory below. Defensive defaults keep things working when the
        ``paths`` block is omitted from the YAML.

        The held-out test split may live at a non-standard location (e.g. the
        MIT-MI test scenes ship under ``test/input/test`` + ``test/gt/test``
        rather than ``jpg/test`` + ``jpg_gt/test``). When ``split == 'test'``
        and ``paths.test_jpg_root`` / ``paths.test_gt_root`` are set, those are
        used verbatim (they already point at the scene-parent dir); otherwise we
        fall back to the standard ``<jpg_root>/test`` layout.
        """
        paths = self.cfg.get("paths", {})
        if split == "test" and paths.get("test_jpg_root") and paths.get("test_gt_root"):
            return (
                Path(paths["test_jpg_root"]),
                Path(paths["test_gt_root"]),
            )
        return (
            Path(paths.get("jpg_root", DEFAULT_JPG_ROOT)) / split,
            Path(paths.get("jpg_gt_root", DEFAULT_JPG_GT_ROOT)) / split,
        )

    def _enumerate_scenes(self, jpg_root: Path) -> list[str]:
        if not jpg_root.exists():
            return []
        return sorted(p.name for p in jpg_root.iterdir() if p.is_dir())

    def _build_datasets(self) -> tuple[DAINetDataset, DAINetDataset]:
        """Build train + val datasets from the `train/` folder only.

        Val is internal-to-training; it uses ``mode='val'`` to keep the
        per-direction (φ, θ) diagnostics + failure visualizations available,
        but the model is still called RGB-only during the val pass — see
        ``_evaluate``.
        """
        jpg_root, jpg_gt_root = self._split_roots("train")
        ds_cfg = self.cfg.get("dataset", {})
        subset = float(ds_cfg.get("subset_ratio", 1.0))
        val_ratio = float(ds_cfg.get("val_ratio", 0.1))

        pool = self._enumerate_scenes(jpg_root)
        pool = subset_scenes(pool, ratio=subset)
        train_scenes, val_scenes = train_val_split(pool, val_ratio=val_ratio)

        augment_cfg = self.cfg.get("augmentation", {})

        # Direction-generalisation split (ablation C1). When `dataset.leave_k_dirs`
        # is set, train on the kept directions and validate on the HELD-OUT
        # directions — so the val number measures generalisation to unseen
        # light directions. Default (no block) = all 25 directions everywhere.
        lkd = ds_cfg.get("leave_k_dirs")
        train_dirs = held_out_dirs = None
        if lkd:
            k = int(lkd.get("k", 5))
            seed = int(lkd.get("seed", 42))
            train_dirs, held_out_dirs = leave_k_directions_out(list(range(25)), k=k, seed=seed)

        # Stash the resolved split so a no-augmentation train-split *eval*
        # dataset can be rebuilt deterministically later (scripts/test.py
        # reports metrics on train + val + test).
        self._train_scenes = train_scenes
        self._train_dirs = train_dirs
        self._ds_common = self._dataset_common_kwargs()

        return (
            DAINetDataset(
                jpg_root,
                jpg_gt_root,
                train_scenes,
                augment=True,
                mode="train",
                augment_cfg=augment_cfg,
                split="train",
                direction_ids=train_dirs,
                **self._ds_common,
            ),
            DAINetDataset(
                jpg_root,
                jpg_gt_root,
                val_scenes,
                augment=False,
                mode="val",
                split="train",  # val scenes still live under train/
                direction_ids=held_out_dirs,
                **self._ds_common,
            ),
        )

    def _dataset_common_kwargs(self) -> dict:
        """Constructor kwargs shared by every train/val DAINetDataset (the roots,
        size, and material-taxonomy knobs that do not vary across splits)."""
        paths = self.cfg.get("paths", {})
        data_cfg = self.cfg.get("data", {})
        return dict(
            size=tuple(self.cfg.get("spatial", {}).get("size", [512, 640])),
            sam_root=paths.get("sam_root", "data/raw/mit_mi/sam_masks"),
            superpixel_root=paths.get("superpixel_root", "data/raw/mit_mi/superpixels"),
            normals_root=paths.get("normals_root", "data/raw/mit_mi/normals"),
            material_taxonomy_path=data_cfg.get(
                "material_taxonomy_path", "data/raw/mit_mi/material_taxonomy.json"
            ),
            use_material=bool(data_cfg.get("use_material", True)) and self._k_material > 0,
        )

    def build_train_eval_dataset(self) -> DAINetDataset | None:
        """The train scenes as a no-augmentation, probe-masked *eval* dataset.

        Mirrors the val dataset (``augment=False``, ``mode='val'`` → probe mask
        + per-direction diagnostics + RGB-only eval contract) but over the
        train scene list, so ``scripts/test.py`` can report a train-fit number
        alongside val and test. Returns ``None`` if the split was never
        resolved (e.g. datasets were injected rather than built)."""
        if not getattr(self, "_train_scenes", None):
            return None
        jpg_root, jpg_gt_root = self._split_roots("train")
        return DAINetDataset(
            jpg_root,
            jpg_gt_root,
            self._train_scenes,
            augment=False,
            mode="val",
            split="train",
            direction_ids=self._train_dirs,
            **self._dataset_common_kwargs(),
        )

    def build_test_dataset(self) -> DAINetDataset:
        """Held-out test dataset under the single-RGB-input contract.

        Reads the test split from ``paths.test_jpg_root`` / ``paths.test_gt_root``
        when set, else ``<jpg_root>/test``. Uses ``mode='test'`` — the model
        receives only the sRGB image, but the dataset attaches the priors it
        computes from that RGB at full capacity: SAM2 ids (precomputed for the
        test split through the FROZEN train centroids) and DSINE normals. Each
        scene's ``meta.json`` supplies the chrome+gray probe mask (metrics-only)
        plus (φ,θ,b) for diagnostics — never fed to the model (the direction head
        predicts direction). SAM/normals roots come from config so the test split
        resolves the same cache locations the precompute scripts write.
        """
        jpg_root, jpg_gt_root = self._split_roots("test")
        ds_cfg = self.cfg.get("dataset", {})
        paths = self.cfg.get("paths", {})
        subset = float(ds_cfg.get("subset_ratio", 1.0))
        pool = self._enumerate_scenes(jpg_root)
        pool = subset_scenes(pool, ratio=subset)
        size = tuple(self.cfg.get("spatial", {}).get("size", [512, 640]))
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

    # ------------------------------------------------------------- helpers
    def _move_batch(self, batch: dict) -> dict:
        out: dict = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                t = v.to(self.device, non_blocking=True)
                # Only the 4-D image tensors should be channels_last; leaving
                # masks / seg ids contiguous keeps PyTorch warnings quiet.
                if (
                    self.channels_last
                    and t.dim() == 4
                    and k in ("input_rgb", "target", "normals")
                ):
                    t = t.contiguous(memory_format=torch.channels_last)
                out[k] = t
            else:
                out[k] = v
        return out

    def _step_lr(self, epoch_frac: float) -> None:
        warmup = self.warmup_epochs
        total = self.epochs
        if epoch_frac < warmup:
            lr = self.base_lr * (epoch_frac / max(warmup, 1))
        else:
            cos_frac = (epoch_frac - warmup) / max(total - warmup, 1)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * min(cos_frac, 1.0))
            )
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def _call_model(self, batch: dict, *, drop_cond: bool) -> dict:
        """Call the model with full or null conditioning.

        Loss-side seg / sh inputs from the batch stay untouched — only the
        model's conditioning is dropped. When the model's illum conditioning
        is dropped, we also zero ``has_sh`` AND ``has_material`` so the
        respective aux losses skip this batch (the model has no direction
        signal to predict SH from, and the null-cond path has no material
        supervision to learn from — same pattern, both are training-only).
        """
        # Normals are *spatial* input (geometry of the scene), not
        # classifier-free droppable conditioning. Pass them on both
        # branches when available so the encoder always sees the same
        # 6-channel input distribution.
        normals = batch.get("normals")
        if drop_cond:
            if "has_sh" in batch and isinstance(batch["has_sh"], torch.Tensor):
                batch["has_sh"] = torch.zeros_like(batch["has_sh"])
            if "has_material" in batch and isinstance(batch["has_material"], torch.Tensor):
                batch["has_material"] = torch.zeros_like(batch["has_material"])
            # No (φ,θ,b) here ⇒ with the direction head on, the model conditions
            # on its OWN predicted direction (matches the inference path).
            return self.model(
                batch["input_rgb"],
                normals=normals,
                compute_material=False,
            )
        # Full-cond branch: teacher-force GT direction most of the time, but on
        # a `direction_pred_p` fraction feed the head's prediction so the model
        # learns to use its own (φ,θ,b) the way it will at inference.
        use_pred_direction = (
            self._use_direction_head
            and self._cond_rng.random() < self.direction_pred_p
        )
        return self.model(
            batch["input_rgb"],
            seg=batch.get("input_seg"),
            phi=batch.get("phi"),
            theta=batch.get("theta"),
            bnorm=batch.get("brightness_norm"),
            direction_id=batch.get("direction_id"),
            normals=normals,
            use_pred_direction=use_pred_direction,
        )

    # --------------------------------------------------------------- eval
    def _evaluate(
        self,
        *,
        desc: str = "Val",
        failure_buckets: FailureBuckets | None = None,
        max_batches: int | None = None,
    ) -> tuple[dict[str, float], dict[str, float], int]:
        """Single-bucket validation pass — model called RGB-only.

        Returns ``(metrics, losses, n_batches)``.
        - ``metrics`` is the flat dict from `MetricComputer.compute()`
          (the five benchmark metrics).
        - ``losses`` is the running mean of every active weighted loss term
          (paired losses generally gate to zero on the unpaired val loader).
        - ``n_batches`` is the number of validation batches processed
          (capped by ``max_batches`` if given, else ``training.val_max_batches``).

        ``max_batches`` overrides ``self.val_max_batches`` when set — used
        by the interval-val call site to apply a tighter cap than the
        epoch-end pass.

        Calling the model RGB-only here aligns the val score with the
        test/benchmark deployment scenario.
        """
        from evaluation.metrics import MetricComputer

        was_training = self.model.training
        self.model.eval()
        with_lpips = True  # the LPIPS *metric* is always on for benchmarking
        metrics = MetricComputer(with_lpips=with_lpips)
        loss_sums: dict[str, float] = defaultdict(float)
        n_batches = 0
        autocast_ctx = (
            torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype)
            if torch.cuda.is_available()
            else torch.amp.autocast("cpu", enabled=False)
        )
        cap = max_batches if max_batches is not None else self.val_max_batches
        loader_len = len(self.val_loader)
        with torch.no_grad(), autocast_ctx:
            for i, batch in enumerate(tqdm(self.val_loader, desc=desc, total=loader_len)):
                if cap is not None and i >= cap:
                    break
                batch = self._move_batch(batch)
                out = self.model(
                    batch["input_rgb"],
                    seg=batch.get("input_seg"),
                    normals=batch.get("normals"),
                    compute_material=True,
                )
                pred = out["output"].float()
                target = batch["target"]
                metrics.update(pred, target, mask=batch.get("probe_mask"))
                try:
                    # Diagnostics (3rd return) are intentionally NOT summed
                    # into loss_sums — the val loss total must aggregate
                    # weighted losses only (see losses/manager.py).
                    _, loss_terms, _diag = self.loss_fn(out, batch)
                    for k, v in loss_terms.items():
                        if torch.isfinite(v).all():
                            loss_sums[k] += float(v.detach().item())
                except Exception:  # noqa: BLE001 — val loss is diagnostic; never block eval
                    pass
                n_batches += 1
                if failure_buckets is not None and "phi" in batch and "theta" in batch:
                    failure_buckets.update(
                        pred,
                        target,
                        phi=batch["phi"],
                        theta=batch["theta"],
                        scenes=batch.get("scene"),
                        input_rgb=batch["input_rgb"],
                    )
        if was_training:
            self.model.train()
        loss_means = {k: v / max(n_batches, 1) for k, v in loss_sums.items()}
        return metrics.compute(), loss_means, n_batches

    def _render_val_samples(self, epoch: int) -> tuple[list, list[str]]:
        """Return a list of `input | prediction` PIL images, one per sample.

        Also writes the legacy grid PNG to `plots/val_samples/` for the
        offline thesis figures. The grid is no longer logged to wandb —
        wandb gets the per-sample images instead so the user can inspect
        each scene individually.
        """
        if not self._val_sample_indices:
            return [], []
        rows = []
        self.model.eval()
        with torch.no_grad():
            for idx in self._val_sample_indices:
                item = self.val_ds[idx]
                inp = item["input_rgb"].unsqueeze(0).to(self.device)
                nm = item.get("normals")
                nm_t = nm.unsqueeze(0).to(self.device) if nm is not None else None
                sg = item.get("input_seg")
                sg_t = sg.unsqueeze(0).to(self.device) if sg is not None else None
                out = self.model(inp, seg=sg_t, normals=nm_t, compute_material=False)
                rows.append(
                    {
                        "input": inp[0].cpu(),
                        "target": item["target"],
                        "pred": out["output"][0].cpu(),
                        "R": out["reflectance"][0].cpu(),
                        "scene": item["scene"],
                        "direction_id": int(item["direction_id"].item()),
                    }
                )
        self.model.train()
        if not rows:
            return [], []
        input_t = torch.stack([r["input"] for r in rows])
        target_t = torch.stack([r["target"] for r in rows])
        pred_t = torch.stack([r["pred"] for r in rows])
        R_t = torch.stack([r["R"] for r in rows])
        captions = [f"{r['scene']} dir{r['direction_id']}" for r in rows]
        # Keep the legacy grid PNG on disk for offline figures.
        out_path = self.plot_dir / "val_samples" / f"epoch_{epoch:03d}.png"
        make_val_samples_grid(
            input_t, target_t, pred_t, R_t, out_path, captions=captions,
            max_samples=len(rows),
        )
        pair_images = make_val_pair_images(input_t, pred_t, captions=captions)
        return pair_images, captions

    def _render_failure_scenes(self, buckets: FailureBuckets, epoch: int) -> list:
        """Per-sample side-by-side `input | prediction` images for worst-K.

        Also writes the [Input|Target|Pred] grid PNG to
        `plots/val_failures/` for offline figures.
        """
        if not buckets.worst_samples:
            return []
        out_path = self.plot_dir / "val_failures" / f"epoch_{epoch:03d}.png"
        make_val_failures_grid(buckets.worst_samples, out_path)
        return make_val_failure_pair_images(buckets.worst_samples)

    def _run_interval_validation(
        self,
        *,
        epoch: int,
        global_step: int,
    ) -> None:
        """Sub-epoch validation hook — runs live + EMA over the full val set
        and emits wandb / JSONL / CSV rows under the same schema as
        per-epoch validation. Called every `val_interval_iters` steps."""
        # Live val — track worst-K so failures section gets per-step samples.
        live_failures = FailureBuckets(
            phi_bins=8, theta_bins=4, worst_top_k=self._n_worst
        )
        val_metrics, val_losses, _ = self._evaluate(
            desc=f"Val[live] step{global_step}",
            failure_buckets=live_failures,
            max_batches=self.val_interval_max_batches,
        )
        val_total = sum(val_losses.values())
        self.wandb.log(
            {
                "loss_total": val_total,
                **{f"loss/{k}": v for k, v in val_losses.items()},
            },
            step=global_step,
            phase="val_live",
        )
        # Section 6 — live val eval metrics (PSNR, MS-SSIM, LPIPS) on wandb.
        self.wandb.log_metrics(val_metrics, phase="val_live", step=global_step)
        # Live val media — per-sample input|prediction images so the user
        # sees prediction snapshots mid-epoch, not only at epoch boundary.
        # Skipped entirely when logging.log_media is false (the default).
        if self.log_media:
            pair_images, _captions = self._render_val_samples(epoch)
            for k, img in enumerate(pair_images):
                self.wandb.log_images(
                    "predictions", {f"sample_{k}": img}, step=global_step,
                )
            failure_pairs = self._render_failure_scenes(live_failures, epoch)
            for k, img in enumerate(failure_pairs):
                self.wandb.log_images(
                    "failures", {f"sample_{k}": img}, step=global_step,
                )
        self._append_iter_jsonl(
            {
                "phase": "val_live",
                "epoch": epoch,
                "step": global_step,
                "losses": dict(val_losses),
                "metrics": dict(val_metrics),
            }
        )
        self._append_metrics_csv(
            {
                "event": "interval",
                "phase": "val_live",
                "epoch": epoch,
                "global_step": global_step,
                **val_metrics,
                "loss_total": val_total,
            }
        )
        # EMA val.
        if self.ema_eval and self.ema is not None:
            with self.ema.average_parameters(self.model):
                ema_metrics, ema_losses, _ = self._evaluate(
                    desc=f"Val[ema]  step{global_step}",
                    max_batches=self.val_interval_max_batches,
                )
            ema_total = sum(ema_losses.values())
            # Section 6 — EMA val eval metrics only (no EMA loss section in
            # the spec). ema_total still feeds the CSV / JSONL below.
            self.wandb.log_metrics(ema_metrics, phase="val_ema", step=global_step)
            self._append_iter_jsonl(
                {
                    "phase": "val_ema",
                    "epoch": epoch,
                    "step": global_step,
                    "losses": dict(ema_losses),
                    "metrics": dict(ema_metrics),
                }
            )
            self._append_metrics_csv(
                {
                    "event": "interval",
                    "phase": "val_ema",
                    "epoch": epoch,
                    "global_step": global_step,
                    **ema_metrics,
                    "loss_total": ema_total,
                }
            )

    # ------------------------------------------------------- JSON / CSV logging
    def _append_iter_jsonl(self, entry: dict) -> None:
        """Append one structured JSON line to logs/iter_history.jsonl.

        Open-append-close per write so a mid-run crash leaves a valid prefix.
        Called at every `log_interval` for train rows and every interval-val
        for val rows.
        """
        with self._iter_jsonl_path.open("a") as fh:
            fh.write(json.dumps(entry) + "\n")

    def _append_metrics_csv(self, row: dict) -> None:
        """Append one validation row to logs/metrics_history.csv.

        Columns (in order): event, phase, epoch, global_step,
        psnr, ms_ssim, lpips, loss_total.

        Header is written exactly once at the first call. The CSV is the
        durable artifact for thesis-table import (pandas / Excel); the
        JSONL is the structured-log source of truth.
        """
        columns = [
            "event",
            "phase",
            "epoch",
            "global_step",
            "psnr",
            "ms_ssim",
            "lpips",
            "loss_total",
        ]
        write_header = not self._metrics_csv_header_written
        with self._metrics_csv_path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            if write_header:
                writer.writeheader()
                self._metrics_csv_header_written = True
            writer.writerow({c: row.get(c, "") for c in columns})

    def _write_epoch_summary_atomic(self) -> None:
        """Atomically rewrite logs/epoch_summary.json with every completed epoch.

        Write to .tmp then `os.replace` so a crash mid-write never produces
        truncated JSON. The full summary is rewritten each time (small, one
        entry per epoch) so the file can be opened standalone offline.
        """
        run_name = None
        if self.wandb.enabled and self.wandb.run is not None:
            run_name = getattr(self.wandb.run, "name", None)
        body = {
            "run_name": run_name,
            "epochs": self._epoch_summaries,
        }
        tmp_path = self._epoch_summary_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(body, indent=2))
        os.replace(tmp_path, self._epoch_summary_path)

    # ---------------------------------------------------------------- ckpt
    def _write_iter_checkpoint(self, epoch: int, step: int, batch_in_epoch: int = 0) -> None:
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scaler": self.scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                # Batches already trained in `epoch` at save time. Lets a
                # mid-epoch resume fast-forward to exactly here instead of
                # restarting the epoch at 0% (0 = clean epoch boundary).
                "batch_in_epoch": int(batch_in_epoch),
                "config": self.cfg,
                # Full state for a faithful resume.
                "ema_state": self.ema.state_dict() if self.ema is not None else None,
                "best_val": self._best_val,
                "best_epoch": self._best_epoch,
                "stalls": self._stalls,
                "wandb_id": getattr(self.wandb, "run_id", None),
            },
            self.ckpt_dir / "latest.pt",
        )

    @staticmethod
    def _git_sha_now() -> str | None:
        import subprocess

        try:
            return (
                subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=str(Path(__file__).resolve().parent.parent),
                    stderr=subprocess.DEVNULL,
                )
                .decode()
                .strip()
            )
        except Exception:  # noqa: BLE001 — provenance is best-effort
            return None

    @staticmethod
    def _utc_now_iso() -> str:
        import datetime

        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def _cpu_state(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def _save_report_checkpoint(
        self, path: Path, *, epoch: int, val_metrics: dict | None, use_ema: bool
    ) -> str:
        """Write model_best / model_final with BOTH weight sets + provenance.

        ``model`` holds the SELECTED-best weights (EMA if it won selection, else
        live) so readers that do ``ckpt["model"]`` (test.py / make_report /
        infer) still load the best weights. ``model_live`` and ``ema`` retain
        each variant so the run reproduces offline with **zero retraining**.
        Returns the ``selected_by`` tag.
        """
        live = self._cpu_state()
        ema_state = None
        if self.ema is not None:
            with self.ema.average_parameters(self.model):
                ema_state = self._cpu_state()
        selected_by = "ema" if (use_ema and ema_state is not None) else "live"
        selected = ema_state if selected_by == "ema" else live
        save_checkpoint(
            {
                "model": selected,        # best weights (back-compat readers)
                "model_live": live,       # raw live weights
                "ema": ema_state,         # EMA-averaged weights (None if no EMA)
                "selected_by": selected_by,
                "config": self.cfg,
                "epoch": int(epoch),
                "best_val": self._best_val,
                "best_metric": self.es_metric,
                "val_metrics": dict(val_metrics) if val_metrics else None,
                "git_sha": self._git_sha,
                "timestamp": self._utc_now_iso(),
            },
            path,
        )
        return selected_by

    def _write_run_meta(self, *, steps: int) -> None:
        """Self-describing per-run record so the run is reproducible offline."""
        import hashlib

        import yaml as _yaml

        cfg_sha = hashlib.sha256(
            _yaml.safe_dump(self.cfg, sort_keys=True).encode()
        ).hexdigest()
        run_name = self.cfg.get("wandb", {}).get("run_name") or self.cfg.get(
            "_experiment", {}
        ).get("name")
        meta = {
            "run_name": run_name,
            "git_sha": self._git_sha,
            "config_sha256": cfg_sha,
            "best_epoch": self._best_epoch,
            "best_val": self._best_val,
            "best_metric": self.es_metric,
            "best_selected_by": self._best_selected_by,
            "total_steps": steps,
            "epochs": self.epochs,
            "timestamp": self._utc_now_iso(),
            "checkpoints": {
                "best": str(self.ckpt_dir / "model_best.pt"),
                "final": str(self.ckpt_dir / "model_final.pt"),
                "latest": str(self.ckpt_dir / "latest.pt"),
            },
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        (self.log_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # -------------------------------------------------------------- resume
    def _prepare_logs(self, resume: bool) -> None:
        """Decide the jsonl / CSV lifecycle for this fit().

        Fresh run: truncate iter_history.jsonl and start a new CSV header.
        Resume: keep both files (append) and restore the in-memory
        epoch-summary list so epoch_summary.json keeps accumulating instead
        of being overwritten with only the post-resume epochs.
        """
        csv_has_rows = (
            self._metrics_csv_path.exists()
            and self._metrics_csv_path.stat().st_size > 0
        )
        if resume and csv_has_rows:
            self._metrics_csv_header_written = True  # append, don't re-header
            if self._epoch_summary_path.exists():
                try:
                    prior = json.loads(self._epoch_summary_path.read_text())
                    self._epoch_summaries = list(prior.get("epochs", []))
                except Exception:
                    self._epoch_summaries = []
            return
        # Fresh run.
        self._iter_jsonl_path.write_text("")
        self._metrics_csv_header_written = False
        self._epoch_summaries = []

    def load_resume(self, ckpt_path: str | Path) -> tuple[int, int, int]:
        """Restore model / optimizer / scaler / EMA / best-tracker from a
        ``latest.pt`` checkpoint and return ``(start_epoch, start_step,
        batch_offset)`` to continue ``fit()``. ``batch_offset`` is how many
        batches of ``start_epoch`` were already trained, so fit() resumes
        mid-epoch instead of restarting it at 0%. Tolerant of older checkpoints
        that predate the ema_state / wandb_id / best_* / batch_in_epoch fields.
        """
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.use_scaler and ckpt.get("scaler"):
            try:
                self.scaler.load_state_dict(ckpt["scaler"])
            except Exception:
                pass
        # EMA: load the saved shadow, else re-seed from the resumed live
        # weights — an old checkpoint has no ema_state, and seeding from the
        # random init left in __init__ would corrupt the EMA val pass.
        if self.ema is not None:
            if ckpt.get("ema_state"):
                self.ema.load_state_dict(ckpt["ema_state"])
            else:
                self.ema = EMAModel(self.model, decay=self.ema_decay)
        # Best-checkpoint tracker so we don't clobber a better model_best.pt.
        if "best_val" in ckpt:
            self._best_val = float(ckpt["best_val"])
        if "best_epoch" in ckpt:
            self._best_epoch = int(ckpt["best_epoch"])
        if "stalls" in ckpt:
            self._stalls = int(ckpt["stalls"])
        self._resume_wandb_id = ckpt.get("wandb_id")
        start_step = int(ckpt.get("step", 0))
        iters_per_epoch = max(len(self.train_loader), 1)
        ckpt_epoch = int(ckpt.get("epoch", -1))
        batch_in_epoch = ckpt.get("batch_in_epoch")

        # Durable completed-epoch count from epoch_summary.json. Only a fully
        # finished epoch appends an entry, so this is the authority for how
        # many epochs are *done* — but it does NOT exist until the first epoch
        # completes, so it cannot be the primary signal for mid-epoch resumes.
        completed = None
        if self._epoch_summary_path.exists():
            try:
                completed = len(
                    json.loads(self._epoch_summary_path.read_text()).get("epochs", [])
                )
            except Exception:
                completed = None

        if batch_in_epoch is not None:
            # Modern checkpoint: (epoch, batch_in_epoch) fully describe progress
            # and self-disambiguate, so do NOT add 1 to `epoch` blindly.
            #   0 < batch_in_epoch < iters → mid-epoch snapshot of `epoch`;
            #     resume INTO that epoch and fast-forward the trained batches.
            #   batch_in_epoch == 0 (clean boundary) or >= iters (epoch done) →
            #     `epoch` completed; resume the NEXT epoch from the top.
            bie = int(batch_in_epoch)
            if 0 < bie < iters_per_epoch:
                start_epoch, batch_offset = ckpt_epoch, bie
            else:
                start_epoch, batch_offset = ckpt_epoch + 1, 0
        else:
            # Legacy checkpoint (no batch_in_epoch). Trust the durable epoch
            # summary count if present, else assume `epoch` is the last
            # completed epoch. Derive the offset from global_step ONLY when it
            # is self-consistent — i.e. lands inside the resumed epoch's
            # [base, base+iters) window — else restart the epoch (offset 0) to
            # avoid skipping untrained batches.
            start_epoch = completed if completed else ckpt_epoch + 1
            raw = start_step - start_epoch * iters_per_epoch
            batch_offset = raw if 0 <= raw < iters_per_epoch else 0

        # Floor: never resume before an epoch the summary already recorded as
        # fully complete (defends against a stale checkpoint older than the
        # summary). A mid-epoch resume where completed == start_epoch is the
        # normal consistent case and must keep its offset, hence strict `>`.
        if completed is not None and completed > start_epoch:
            start_epoch, batch_offset = completed, 0

        batch_offset %= iters_per_epoch
        print(
            f"[dainet resume] {ckpt_path}: start_epoch={start_epoch} "
            f"start_step={start_step} batch_offset={batch_offset}/{iters_per_epoch} "
            f"best_val={self._best_val:.4f}"
        )
        return start_epoch, start_step, batch_offset

    # ----------------------------------------------------------------- fit
    def _print_run_banner(self, epochs: int) -> None:
        """Detailed run-start banner — printed once before the epoch loop.

        Surfaces data split sizes, batch sizes, AMP / channels-last mode,
        and optimiser/EMA settings so any captured stdout makes the run
        self-describing without having to cross-reference the YAML.
        """
        train_cfg = self.cfg.get("training", {})
        opt_cfg = train_cfg.get("optimizer", {})
        ds_cfg = self.cfg.get("dataset", {})
        size = self.cfg.get("spatial", {}).get("size", [512, 640])
        amp_mode = str(train_cfg.get("amp", "bf16")).lower()
        n_train = len(getattr(self.train_ds, "samples", self.train_ds))
        n_val = len(getattr(self.val_ds, "samples", self.val_ds))
        try:
            n_train_scenes = len({s[0] for s in self.train_ds.samples})
            n_val_scenes = len({s[0] for s in self.val_ds.samples})
        except Exception:  # noqa: BLE001 — dataset shape is informational only
            n_train_scenes = n_val_scenes = -1
        iters_per_epoch = len(self.train_loader)
        val_cap = self.val_max_batches if self.val_max_batches is not None else "full"
        material_flag = bool(self.cfg.get("data", {}).get("use_material", True)) and self._k_material > 0
        print(
            "[dainet run] dataset=MIT-Multi-Illumination split=train/val (held-out test untouched)\n"
            f"           train_scenes={n_train_scenes}  val_scenes={n_val_scenes}  "
            f"train_samples={n_train}  val_samples={n_val}  subset_ratio={ds_cfg.get('subset_ratio', 1.0)}\n"
            f"           batch_size={self.batch_size}  val_batch_size={train_cfg.get('val_batch_size', '?')}  "
            f"val_max_batches={val_cap}  iters/epoch={iters_per_epoch}  epochs={epochs}\n"
            f"           image_size={tuple(size)}  amp={amp_mode}  channels_last={self.channels_last}  "
            f"grad_accum={self.grad_accum}  grad_clip={self.grad_clip}  null_cond_prob={self.null_cond_prob}\n"
            f"           optimizer={opt_cfg.get('name', 'adamw')}  lr={self.base_lr}  "
            f"weight_decay={opt_cfg.get('weight_decay', 0.01)}  min_lr={self.min_lr}  "
            f"warmup_epochs={self.warmup_epochs}\n"
            f"           ema_decay={self.ema_decay}  ema_eval={self.ema_eval}  "
            f"ema_eval_every_n={self.ema_eval_every_n}  log_interval={self.log_interval}  "
            f"ckpt_interval_iters={self.ckpt_interval}\n"
            f"           material_supervision={material_flag}  k_material={self._k_material}\n"
            f"           loss_weights={dict(self.cfg.get('loss', {}))}",
            flush=True,
        )

    def fit(
        self,
        max_epochs: int | None = None,
        max_steps: int | None = None,
        *,
        start_epoch: int = 0,
        start_step: int = 0,
        batch_offset: int = 0,
        resume: bool = False,
    ) -> dict:
        self._prepare_logs(resume)
        global_step = start_step
        epochs = max_epochs if max_epochs is not None else self.epochs
        stop = False
        self._print_run_banner(epochs)
        # On resume, advance the batch sampler's per-epoch RNG so the resumed
        # epoch is shuffled the same way a fresh run would shuffle it — then
        # fast-forwarding `batch_offset` batches lands on the genuinely
        # remaining samples. Without this the sampler restarts at _epoch=0 and
        # the resumed epoch would see a different permutation.
        if resume:
            bs = getattr(self.train_loader, "batch_sampler", None)
            if bs is not None and hasattr(bs, "_epoch"):
                bs._epoch = start_epoch
        for epoch in range(start_epoch, epochs):
            self.model.train()
            # Notify the loss manager of the current epoch so any epoch-
            # based scheduled weights interpolate correctly.
            if hasattr(self.loss_fn, "set_epoch"):
                self.loss_fn.set_epoch(epoch)
            running: dict[str, float] = defaultdict(float)
            running_count = 0
            t0 = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            pbar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch+1}/{epochs}",
                dynamic_ncols=True,
            )
            # Defer .item() sync: accumulate loss tensors and only convert
            # to python floats at log_interval.
            interval_tensors: dict[str, torch.Tensor] = {}
            interval_grad_norms: list[float] = []
            interval_count = 0
            # Per-iter timing — host-side `iter_time`/`data_time` are launched
            # cheaply; `fwd_time`/`bwd_time`/`opt_time` use CUDA Events whose
            # elapsed_time is only read once per log_interval (single sync,
            # same cadence as the loss-tensor .item() sync above).
            interval_iter_times: list[float] = []
            interval_data_times: list[float] = []
            interval_events: list[tuple] = []
            last_iter_end_t = time.perf_counter()
            cuda_for_timing = torch.cuda.is_available()
            # Mid-epoch resume: skip the batches already trained in this epoch
            # before the checkpoint. Only the first resumed epoch is partial.
            skip_batches = batch_offset if (resume and epoch == start_epoch) else 0
            if skip_batches:
                print(
                    f"[dainet resume] fast-forwarding {skip_batches}/{len(self.train_loader)} "
                    f"batches into epoch {epoch+1} (already trained before the kill)",
                    flush=True,
                )
            for batch_idx, batch in enumerate(pbar):
                if batch_idx < skip_batches:
                    # Already-trained batch: advance the loader/RNG without any
                    # compute or step/global_step increment. Keep the timing
                    # anchor fresh so the first real iter's data_time is sane.
                    last_iter_end_t = time.perf_counter()
                    continue
                iter_t0 = time.perf_counter()
                data_time_s = iter_t0 - last_iter_end_t
                batch = self._move_batch(batch)
                self._step_lr(epoch + batch_idx / max(len(self.train_loader), 1))

                drop_cond = self._cond_rng.random() < self.null_cond_prob

                if cuda_for_timing:
                    e_fwd_start = torch.cuda.Event(enable_timing=True)
                    e_fwd_end = torch.cuda.Event(enable_timing=True)
                    e_bwd_end = torch.cuda.Event(enable_timing=True)
                    e_opt_end = torch.cuda.Event(enable_timing=True)
                    e_fwd_start.record()
                with torch.amp.autocast("cuda", enabled=self.amp_enabled, dtype=self.amp_dtype):
                    out = self._call_model(batch, drop_cond=drop_cond)
                # Loss compute is always fp32. Cast float tensors in `out`
                # so the gradient graph re-enters fp32 here — model forward
                # ran under autocast (bf16 by default) for speed, but every
                # sqrt / log / atan2 in the loss stack runs in fp32.
                if cuda_for_timing:
                    e_fwd_end.record()
                if self.amp_enabled:
                    out_loss = {
                        k: (v.float() if isinstance(v, torch.Tensor) and v.is_floating_point() else v)
                        for k, v in out.items()
                    }
                else:
                    out_loss = out
                total_loss, loss_terms, loss_diag = self.loss_fn(out_loss, batch)

                # ---- Hard-fail loss finiteness check ----
                # No silent recovery: any non-finite loss/output is a
                # numerical-stability bug that must be fixed at the source.
                if not torch.isfinite(total_loss).all():
                    bad_terms = [
                        k for k, v in loss_terms.items() if not torch.isfinite(v).all()
                    ]
                    bad_out = [
                        k for k, v in out.items()
                        if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
                    ]
                    raise RuntimeError(
                        f"Non-finite loss at step {global_step} ep{epoch+1}: "
                        f"bad_terms={bad_terms} bad_out={bad_out}. "
                        "Numerical-stability invariant violated — fix the source."
                    )

                scaled = total_loss / max(self.grad_accum, 1)
                if self.use_scaler:
                    self.scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if cuda_for_timing:
                    e_bwd_end.record()

                if (batch_idx + 1) % self.grad_accum == 0:
                    if self.use_scaler:
                        self.scaler.unscale_(self.optimizer)
                    grad_norm_t = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )
                    grad_norm_val = float(grad_norm_t)
                    # ---- Hard-fail gradient finiteness check ----
                    if not math.isfinite(grad_norm_val):
                        bad = [
                            name for name, p in self.model.named_parameters()
                            if p.grad is not None and not torch.isfinite(p.grad).all()
                        ]
                        raise RuntimeError(
                            f"Non-finite gradient at step {global_step} ep{epoch+1}: "
                            f"grad_norm={grad_norm_val}, bad_params={bad[:10]}. "
                            "Numerical-stability invariant violated — fix the source."
                        )
                    if self.use_scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.ema is not None:
                        self.ema.update(self.model)
                    interval_grad_norms.append(grad_norm_val)
                if cuda_for_timing:
                    e_opt_end.record()
                    interval_events.append((e_fwd_start, e_fwd_end, e_bwd_end, e_opt_end))

                # Accumulate detached tensors — finiteness already verified.
                tl_det = total_loss.detach()
                interval_tensors["loss_total"] = (
                    interval_tensors.get("loss_total", torch.zeros_like(tl_det)) + tl_det
                )
                for k, v in loss_terms.items():
                    key = f"loss/{k}"
                    vd = v.detach()
                    interval_tensors[key] = (
                        interval_tensors.get(key, torch.zeros_like(vd)) + vd
                    )
                # Diagnostics under a `diag/` prefix — kept out of loss_total
                # and out of the wandb per-term loss panel; written to the
                # JSONL only (instrumentation, not supervision).
                for k, v in loss_diag.items():
                    key = f"diag/{k}"
                    vd = v.detach()
                    interval_tensors[key] = (
                        interval_tensors.get(key, torch.zeros_like(vd)) + vd
                    )
                interval_count += 1
                running_count += 1
                global_step += 1
                # Inform the loss manager of the post-step global step so
                # the step-based warmup envelope (retinex_constraint,
                # xdir_relight) ramps in lockstep.
                if hasattr(self.loss_fn, "set_step"):
                    self.loss_fn.set_step(global_step)

                # Record per-iter wall-clock (host side). The CUDA-event
                # elapsed_time read happens in batch at log_interval below.
                iter_t1 = time.perf_counter()
                interval_iter_times.append(iter_t1 - iter_t0)
                interval_data_times.append(data_time_s)
                last_iter_end_t = iter_t1

                if global_step % self.log_interval == 0 and interval_count > 0:
                    # One sync point per log interval, not per batch.
                    sums = {k: float(v.item()) for k, v in interval_tensors.items()}
                    for k, s in sums.items():
                        running[k] += s  # epoch-wide running sum
                    avg = {k: s / max(interval_count, 1) for k, s in sums.items()}
                    cur_lr = self.optimizer.param_groups[0]["lr"]
                    avg_grad_norm = (
                        sum(interval_grad_norms) / len(interval_grad_norms)
                        if interval_grad_norms
                        else 0.0
                    )
                    # Drain timing — single CUDA sync on the last opt event
                    # then read elapsed_time for every event in the interval.
                    avg_iter_time = (
                        sum(interval_iter_times) / len(interval_iter_times)
                        if interval_iter_times
                        else 0.0
                    )
                    avg_data_time = (
                        sum(interval_data_times) / len(interval_data_times)
                        if interval_data_times
                        else 0.0
                    )
                    avg_fwd_time = avg_bwd_time = avg_opt_time = 0.0
                    if cuda_for_timing and interval_events:
                        # Sync on the last opt_end event in this interval.
                        interval_events[-1][3].synchronize()
                        fwd_times = [
                            evs[0].elapsed_time(evs[1]) / 1000.0 for evs in interval_events
                        ]
                        bwd_times = [
                            evs[1].elapsed_time(evs[2]) / 1000.0 for evs in interval_events
                        ]
                        opt_times = [
                            evs[2].elapsed_time(evs[3]) / 1000.0 for evs in interval_events
                        ]
                        avg_fwd_time = sum(fwd_times) / len(fwd_times)
                        avg_bwd_time = sum(bwd_times) / len(bwd_times)
                        avg_opt_time = sum(opt_times) / len(opt_times)
                    avg["lr"] = cur_lr
                    avg["epoch"] = epoch
                    avg["grad_norm"] = avg_grad_norm
                    # Phase-routed wandb log (lands in train/loss/<term>).
                    self.wandb.log(avg, step=global_step, phase="train")
                    # Structured per-iter JSONL — flat losses dict (matches
                    # the new umbrella schema; no category grouping).
                    flat_losses = {
                        k.split("/", 1)[1]: v
                        for k, v in avg.items()
                        if k.startswith("loss/")
                    }
                    flat_diag = {
                        k.split("/", 1)[1]: v
                        for k, v in avg.items()
                        if k.startswith("diag/")
                    }
                    self._append_iter_jsonl(
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "step": global_step,
                            "loss_total": avg["loss_total"],
                            "losses": dict(flat_losses),
                            "diagnostics": dict(flat_diag),
                            "lr": cur_lr,
                            "grad_norm": avg_grad_norm,
                            "iter_time_s": avg_iter_time,
                            "data_time_s": avg_data_time,
                            "fwd_time_s": avg_fwd_time,
                            "bwd_time_s": avg_bwd_time,
                            "opt_time_s": avg_opt_time,
                        }
                    )
                    pbar.set_postfix(
                        loss=avg["loss_total"],
                        grad_norm=avg_grad_norm,
                        it_s=f"{avg_iter_time:.2f}",
                    )
                    # Progress readout (lands in the wandb "Logs" tab via the
                    # console=wrap capture). A bare `step=…` line carried no
                    # sense of how far the epoch is; this reports batch k/N,
                    # percent, throughput and ETA derived from the tqdm bar's
                    # own timing (`format_dict`) so it is correct even when the
                    # process is piped (tmux / nohup / tee) and the live bar
                    # itself does not render.
                    fmt = pbar.format_dict
                    n_done = fmt.get("n", batch_idx + 1)
                    n_total = fmt.get("total") or len(self.train_loader)
                    elapsed = fmt.get("elapsed", 0.0) or 0.0
                    rate = fmt.get("rate")
                    if not rate and elapsed > 0:
                        rate = n_done / elapsed
                    pct = (100.0 * n_done / n_total) if n_total else 0.0
                    eta = tqdm.format_interval(
                        (n_total - n_done) / rate if (rate and n_total) else 0.0
                    )
                    rate_str = f"{rate:.2f}" if rate else "?"
                    tqdm.write(
                        f"[train] ep {epoch+1}/{epochs} {n_done}/{n_total} "
                        f"({pct:.0f}%) | loss {avg['loss_total']:.4f} | "
                        f"lr {avg['lr']:.2e} | gnorm {avg_grad_norm:.2f} | "
                        f"{rate_str} it/s | eta {eta}"
                    )
                    interval_tensors = {}
                    interval_grad_norms = []
                    interval_count = 0
                    interval_iter_times = []
                    interval_data_times = []
                    interval_events = []

                # Per-iteration checkpoint — atomic rewrite of latest.pt.
                # Record batch_idx+1 (batches done this epoch) so a resume
                # picks up mid-epoch instead of restarting at 0%.
                if self.ckpt_interval > 0 and global_step % self.ckpt_interval == 0:
                    self._write_iter_checkpoint(epoch, global_step, batch_idx + 1)

                # Interval validation (live + EMA) — fires every
                # `val_interval_iters` steps so the val curve has sub-epoch
                # resolution. Per-epoch validation still runs at end-of-epoch.
                if (
                    self.val_interval_iters > 0
                    and global_step % self.val_interval_iters == 0
                ):
                    self._run_interval_validation(
                        epoch=epoch,
                        global_step=global_step,
                    )

                if max_steps is not None and global_step >= max_steps:
                    stop = True
                    break

            # Drain trailing tensors not flushed at the last log_interval.
            if interval_tensors:
                for k, v in interval_tensors.items():
                    running[k] += float(v.item())
                interval_tensors = {}
                interval_count = 0

            # Mark end-of-train wall-clock before the live-val pass so we
            # can report train-vs-val time separately.
            train_time_s = time.time() - t0

            # End-of-epoch validation — live weights, single umbrella bucket.
            failure_buckets = FailureBuckets(
                phi_bins=8, theta_bins=4, worst_top_k=self._n_worst
            )
            t_live_val = time.time()
            val_metrics, val_losses, n_val_batches_live = self._evaluate(
                desc=f"Val[live] ep{epoch+1}",
                failure_buckets=failure_buckets,
            )
            live_val_time_s = time.time() - t_live_val
            epoch_time = time.time() - t0
            val_total = sum(val_losses.values())
            self.wandb.log(
                {
                    "loss_total": val_total,
                    **{f"loss/{k}": v for k, v in val_losses.items()},
                },
                step=global_step,
                phase="val_live",
            )
            # Section 5 — live val eval metrics on wandb.
            self.wandb.log_metrics(val_metrics, phase="val_live", step=global_step)
            # Per-epoch media: per-sample input|prediction images under
            # media/predictions/sample_<k> and media/failures/sample_<k>.
            # Skipped entirely when logging.log_media is false (the default) —
            # the render itself (not just the upload) is what costs.
            if self.log_media:
                pair_images, _captions = self._render_val_samples(epoch)
                for k, img in enumerate(pair_images):
                    self.wandb.log_images(
                        "predictions", {f"sample_{k}": img}, step=global_step,
                    )
                failure_pairs = self._render_failure_scenes(failure_buckets, epoch)
                for k, img in enumerate(failure_pairs):
                    self.wandb.log_images(
                        "failures", {f"sample_{k}": img}, step=global_step,
                    )

            # EMA validation — same loop, weights swapped via context manager.
            ema_metrics: dict[str, float] = {}
            ema_losses: dict[str, float] = {}
            ema_val_time_s = 0.0
            n_val_batches_ema = 0
            run_ema_now = (
                self.ema_eval
                and self.ema is not None
                and self.ema_eval_every_n > 0
                and ((epoch + 1) % self.ema_eval_every_n == 0)
            )
            if run_ema_now:
                t_ema_val = time.time()
                with self.ema.average_parameters(self.model):
                    ema_metrics, ema_losses, n_val_batches_ema = self._evaluate(
                        desc=f"Val[ema]  ep{epoch+1}",
                    )
                ema_val_time_s = time.time() - t_ema_val
                epoch_time = time.time() - t0
                ema_total = sum(ema_losses.values())
                # Section 6 — EMA val eval metrics only. ema_total → CSV/JSONL.
                self.wandb.log_metrics(ema_metrics, phase="val_ema", step=global_step)

            # Per-epoch CSV row for thesis-table import.
            self._append_metrics_csv(
                {
                    "event": "epoch",
                    "phase": "val_live",
                    "epoch": epoch,
                    "global_step": global_step,
                    **val_metrics,
                    "loss_total": val_total,
                }
            )
            if run_ema_now:
                self._append_metrics_csv(
                    {
                        "event": "epoch",
                        "phase": "val_ema",
                        "epoch": epoch,
                        "global_step": global_step,
                        **ema_metrics,
                        "loss_total": ema_total,
                    }
                )

            # Per-iter JSONL val rows (matches train rows' structure).
            self._append_iter_jsonl(
                {
                    "phase": "val_live",
                    "epoch": epoch,
                    "step": global_step,
                    "losses": dict(val_losses),
                    "metrics": dict(val_metrics),
                }
            )
            if run_ema_now:
                self._append_iter_jsonl(
                    {
                        "phase": "val_ema",
                        "epoch": epoch,
                        "step": global_step,
                        "losses": dict(ema_losses),
                        "metrics": dict(ema_metrics),
                    }
                )

            # Build the epoch summary entry (single source of truth on disk,
            # flat schema — no metrics_by_bucket).
            train_loss_total_avg = (
                running["loss_total"] / max(running_count, 1) if running_count else 0.0
            )
            train_losses_flat = {
                k.split("/", 1)[1]: (v / max(running_count, 1))
                for k, v in running.items()
                if k.startswith("loss/")
            }
            epoch_entry = {
                "epoch": epoch,
                "global_step": global_step,
                "time_s": epoch_time,
                "train_time_s": train_time_s,
                "live_val_time_s": live_val_time_s,
                "ema_val_time_s": ema_val_time_s,
                "n_val_batches_live": n_val_batches_live,
                "n_val_batches_ema": n_val_batches_ema,
                "train": {
                    "loss_total_avg": train_loss_total_avg,
                    "losses": dict(train_losses_flat),
                    "lr_end": self.optimizer.param_groups[0]["lr"],
                },
                "val_live": {
                    "losses": dict(val_losses),
                    "metrics": dict(val_metrics),
                },
                "val_ema": (
                    {
                        "losses": dict(ema_losses),
                        "metrics": dict(ema_metrics),
                    }
                    if run_ema_now
                    else None
                ),
            }
            self._epoch_summaries.append(epoch_entry)
            self._write_epoch_summary_atomic()

            # Terminal summary — multi-line, self-describing.
            train_losses_str = " ".join(
                f"{k}={v:.4f}" for k, v in epoch_entry["train"]["losses"].items()
            )
            def _val_line(m: dict[str, float]) -> str:
                return (
                    f"PSNR={m.get('psnr', float('nan')):.3f}  "
                    f"MS-SSIM={m.get('ms_ssim', float('nan')):.4f}  "
                    f"LPIPS={m.get('lpips', float('nan')):.4f}"
                )
            live_line = _val_line(val_metrics) + f"  (n_batches={n_val_batches_live})"
            ema_line = (
                _val_line(ema_metrics) + f"  (n_batches={n_val_batches_ema})"
                if run_ema_now
                else "skipped (off-cycle)"
            )
            print(
                f"[dainet epoch {epoch+1}/{epochs}] time={epoch_time:.1f}s  "
                f"(train={train_time_s:.1f}s  val_live={live_val_time_s:.1f}s  "
                f"val_ema={ema_val_time_s:.1f}s)\n"
                f"    train: loss_total={train_loss_total_avg:.4f}  {train_losses_str}\n"
                f"    val_live: {live_line}\n"
                f"    val_ema:  {ema_line}\n"
                f"    lr_end={self.optimizer.param_groups[0]['lr']:.3e}",
                flush=True,
            )

            # End-of-epoch checkpoint write (so a clean epoch boundary always
            # has the freshest latest.pt even if the iter-cadence didn't tick).
            # batch_in_epoch=0: this epoch is in epoch_summary now, so a resume
            # starts the NEXT epoch fresh.
            self._write_iter_checkpoint(epoch, global_step, 0)

            # Pick the better of live vs EMA for best-checkpoint selection.
            # Missing-metric sentinel is the worst possible value for the mode
            # (+inf when lower-is-better, -inf when higher-is-better) so a run
            # without the metric never wins selection.
            worst = float("inf") if self.es_mode == "min" else -float("inf")
            cur_live = val_metrics.get(self.es_metric, worst)
            cur_ema = ema_metrics.get(self.es_metric, worst) if ema_metrics else worst
            ema_is_better = (
                cur_ema < cur_live if self.es_mode == "min" else cur_ema > cur_live
            )
            if ema_is_better and self.ema_eval:
                cur, use_ema = cur_ema, True
            else:
                cur, use_ema = cur_live, False
            improved = (
                cur < self._best_val - self.es_min_delta
                if self.es_mode == "min"
                else cur > self._best_val + self.es_min_delta
            )
            if improved:
                self._best_val = cur
                self._best_epoch = epoch
                self._stalls = 0
                self._best_selected_by = self._save_report_checkpoint(
                    self.ckpt_dir / "model_best.pt",
                    epoch=epoch,
                    val_metrics=(ema_metrics if use_ema else val_metrics),
                    use_ema=use_ema,
                )
            else:
                self._stalls += 1

            if self.es_enabled and self._stalls > self.es_patience:
                break
            if stop:
                break

        # Write model_final with both weight sets + provenance (the "model" key
        # prefers EMA when EMA-eval is on — it generally generalises better).
        self._save_report_checkpoint(
            self.ckpt_dir / "model_final.pt",
            epoch=epoch,
            val_metrics=None,
            use_ema=bool(self.ema is not None and self.ema_eval),
        )
        # No post-training plotting: the durable artifacts are
        # `logs/iter_history.jsonl`, `logs/epoch_summary.json`,
        # `logs/metrics_history.csv`, and `logs/run_meta.json`. Paper figures
        # regenerate offline from the saved checkpoints + these logs.
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._write_run_meta(steps=global_step)
        self.wandb.finish()
        return {"best_val": self._best_val, "best_epoch": self._best_epoch, "steps": global_step}
