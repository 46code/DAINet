"""Shared inference helpers used by `scripts/infer.py` and `scripts/make_pair.py`.

Centralises (1) image I/O, (2) checkpoint + model construction (including the
material-head strip), (3) the SAM2-segmentation + network forward at the
configured network resolution, and (4) up-sampling the prediction back to the
input's native resolution so saved outputs match the input's dimensions.

Keeping these helpers in one place lets `infer.py` (single output per input)
and `make_pair.py` (one before/after pair per input) stay narrowly focused on
their own output format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from data.segmentation import segment_image
from models.network import DAINet


# Lazily-loaded DSINE normal estimator (reuses the precompute wrapper so the
# inference-time normals match the training-time cache exactly). None until the
# first request; "unavailable" if DSINE could not be loaded (→ zero-normals).
_DSINE_ESTIMATOR: Callable[[np.ndarray], np.ndarray] | None = None
_DSINE_TRIED = False


def _get_dsine_estimator(device: str | None = None):
    """Load the DSINE estimator once; return None if it cannot be loaded.

    ``device`` pins the estimator to the inference GPU so DSINE runs on the
    same card as the model and SAM2 (rather than defaulting to cuda:0).
    """
    global _DSINE_ESTIMATOR, _DSINE_TRIED
    if _DSINE_TRIED:
        return _DSINE_ESTIMATOR
    _DSINE_TRIED = True
    try:
        from scripts.precompute_normals import _load_dsine

        _DSINE_ESTIMATOR = _load_dsine(device=device)
    except Exception as exc:  # noqa: BLE001
        print(
            f"[dainet infer] WARNING: DSINE normals unavailable ({exc!r}); "
            "running with zero-normals (degraded — install DSINE to close the "
            "train/deploy gap, see docs/cli.md).",
            flush=True,
        )
        _DSINE_ESTIMATOR = None
    return _DSINE_ESTIMATOR


def _gray_world_correct(pred: np.ndarray, strength: float = 0.6) -> np.ndarray:
    """Residual gray-world white-balance on the prediction (opt-in prior).

    Estimates the leftover global illuminant of the corrected image (its mean
    per-channel RGB) and scales each channel toward the achromatic mean by
    ``strength`` ∈ [0, 1]. A coarse, single-image color-constancy refinement
    layered on top of the network output — documented as approximate.
    """
    mean = pred.reshape(-1, 3).mean(axis=0) + 1e-6
    gray = float(mean.mean())
    gain = (gray / mean)  # per-channel gain toward gray
    gain = 1.0 + strength * (gain - 1.0)
    return np.clip(pred * gain[None, None, :], 0.0, 1.0)


def load_image_rgb(path: Path) -> np.ndarray:
    """Read an sRGB image from disk as uint8 RGB [H, W, 3]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_image_rgb(arr: np.ndarray, path: Path) -> None:
    """Write a float [0,1] RGB image to disk as 8-bit PNG."""
    bgr = cv2.cvtColor(np.clip(arr * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def load_model_from_checkpoint(ckpt_path: str | Path, device: str) -> tuple[DAINet, dict]:
    """Load a dainet checkpoint and instantiate `DAINet` from its saved config.

    The training-only material head is always instantiated as disabled at
    inference (its weights are stripped from the checkpoint state dict before
    `load_state_dict`). Returns ``(model, cfg)`` so callers can also access
    the saved config (e.g. for default network resolution).
    """
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    model = DAINet(
        embed_dim=model_cfg.get("embed_dim", 128),
        seg_embed_dim=model_cfg.get("seg_embed_dim", 128),
        illum_hidden=model_cfg.get("illum_hidden", 128),
        attn_heads=model_cfg.get("attn_heads", 8),
        sh_l_max=model_cfg.get("sh_l_max", 4),
        pretrained_encoder=False,  # weights come from the checkpoint
        backbone=model_cfg.get("backbone", "convnext_tiny"),
        use_swin_bottleneck=model_cfg.get("use_swin_bottleneck", False),
        use_specular_head=model_cfg.get("use_specular_head", False),
        use_illum_token=model_cfg.get("use_illum_token", False),
        use_reflectance_memory=model_cfg.get("use_reflectance_memory", False),
        # Material head: training-only. Instantiated as disabled here; its
        # checkpoint weights are stripped below.
        use_material_head=False,
        material_num_classes=int(model_cfg.get("material_num_classes", 0)),
        material_head_hidden=int(model_cfg.get("material_head_hidden", 256)),
        use_normals=bool(model_cfg.get("use_normals", False)),
        normals_fusion=model_cfg.get("normals_fusion", None),
        use_sam_conditioning=bool(model_cfg.get("use_sam_conditioning", True)),
        direction_encoding=str(model_cfg.get("direction_encoding", "continuous")),
        num_directions=int(model_cfg.get("num_directions", 25)),
        use_illum_chroma_field=bool(model_cfg.get("use_illum_chroma_field", False)),
        use_illuminant_head=bool(model_cfg.get("use_illuminant_head", False)),
        use_direction_head=bool(model_cfg.get("use_direction_head", False)),
    ).to(device)

    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    if isinstance(state, dict):
        material_keys = [k for k in state.keys() if k.startswith("material_head.")]
        if material_keys:
            state = {k: v for k, v in state.items() if not k.startswith("material_head.")}
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def parse_size_arg(size_arg: str, default: tuple[int, int] = (512, 640)) -> tuple[int, int]:
    """Parse a 'H,W' string into (H, W). Falls back to `default` on empty."""
    if not size_arg:
        return default
    parts = [p.strip() for p in size_arg.split(",")]
    if len(parts) != 2:
        raise ValueError(f"--size must be 'H,W'; got {size_arg!r}")
    return int(parts[0]), int(parts[1])


@torch.no_grad()
def infer_single(
    model: DAINet,
    img_rgb_u8: np.ndarray,
    *,
    net_hw: tuple[int, int],
    device: str,
    sam_weights: str | None,
    normals: np.ndarray | None = None,
    compute_normals: bool = True,
    estimate_illuminant: bool = False,
    illuminant_strength: float = 0.6,
) -> np.ndarray:
    """Run the full inference pipeline on a single uint8 RGB image.

    Steps:
      1. Resize input to the network grid (`net_hw``).
      2. Run SAM2 on the resized image so the segmentation resolution matches.
      3. **Compute DSINE normals inline** (when the model was trained with a
         normals path and ``compute_normals`` is set) so the deploy input
         distribution matches training — closes the zero-normals gap. A
         caller-supplied ``normals`` array overrides; if DSINE is unavailable
         the encoder falls back to zero-normals (degraded, warned once).
      4. Forward the model with `seg=` conditioning (illum left null — the
         model uses its learned null_illum_emb / illum_token).
      5. Optionally apply a coarse gray-world illuminant correction to the
         prediction (``estimate_illuminant``) — an opt-in single-image
         color-constancy refinement.
      6. Up-sample the prediction back to the input's native resolution.

    Returns the corrected image as a float32 RGB array in [0, 1] with the
    **input's original shape** (H0, W0, 3).
    """
    H0, W0 = img_rgb_u8.shape[:2]
    H, W = net_hw
    resized = cv2.resize(img_rgb_u8, (W, H), interpolation=cv2.INTER_LINEAR)
    img_f = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
    rgb_t = torch.from_numpy(img_f).unsqueeze(0).to(device)

    # SAM conditioning is config-driven: a model trained with
    # use_sam_conditioning=False always sees its learned null_seg_emb, so
    # running SAM2 here would only feed an untrained seg encoder — skip it.
    if getattr(model, "use_sam_conditioning", True):
        sam_ids = segment_image(resized, weights_path=sam_weights, device=device)
        seg_t = torch.from_numpy(sam_ids.astype(np.int32))[None, None, ...].to(device)
    else:
        seg_t = None

    # Normals: prefer a caller-supplied map; else compute inline with DSINE if
    # the model uses a normals path. Resize to the network grid.
    if normals is None and compute_normals and getattr(model, "use_normals", False):
        estimator = _get_dsine_estimator(device=device)
        if estimator is not None:
            normals = estimator(resized)  # HxWx3 in [-1, 1] at net grid
    normals_t = None
    if normals is not None:
        n = normals.astype(np.float32)
        if n.shape[:2] != (H, W):
            n = cv2.resize(n, (W, H), interpolation=cv2.INTER_LINEAR)
        normals_t = torch.from_numpy(n.transpose(2, 0, 1)).unsqueeze(0).to(device)

    out = model(rgb_t, seg=seg_t, normals=normals_t, compute_material=False)
    pred = out["output"][0].cpu().permute(1, 2, 0).numpy()
    if estimate_illuminant:
        pred = _gray_world_correct(pred, strength=illuminant_strength)
    if (H0, W0) != (H, W):
        pred = cv2.resize(pred, (W0, H0), interpolation=cv2.INTER_CUBIC)
    return np.clip(pred, 0.0, 1.0)


def resolve_device(gpu: int | None) -> str:
    if gpu is not None and torch.cuda.is_available():
        return f"cuda:{gpu}"
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_inputs(
    single_in: str | None,
    single_out: str | None,
    in_dir: str | None,
    out_dir: str | None,
    default_out_dir: str = "out",
) -> list[tuple[Path, Path]]:
    """Resolve --input/--output vs --input_dir/--output_dir into a flat list."""
    if single_in:
        return [(Path(single_in), Path(single_out or "corrected.png"))]
    if not in_dir:
        raise SystemExit("Need --input or --input_dir.")
    in_root = Path(in_dir)
    out_root = Path(out_dir or default_out_dir)
    extensions = {".jpg", ".jpeg", ".png"}
    pairs: list[tuple[Path, Path]] = []
    for p in sorted(in_root.iterdir()):
        if p.suffix.lower() in extensions:
            # Always write PNG: reusing the input name (e.g. `1.jpeg`) made
            # cv2.imwrite re-encode the corrected output as *lossy JPEG*,
            # silently degrading the very result we are inspecting. Force the
            # lossless `.png` container regardless of the input's extension.
            pairs.append((p, out_root / f"{p.stem}.png"))
    return pairs


def add_prior_args(parser) -> None:
    """Add the shared inference-prior flags to an argparse parser."""
    parser.add_argument(
        "--no_normals", action="store_true",
        help="Disable inline DSINE normals (run with zero-normals — degraded for "
             "normals-trained models; default is to compute normals inline).",
    )
    parser.add_argument(
        "--estimate_illuminant", action="store_true",
        help="Apply a coarse gray-world illuminant correction on top of the "
             "prediction (opt-in single-image color-constancy refinement).",
    )
    parser.add_argument(
        "--illuminant_strength", type=float, default=0.6,
        help="Strength of the --estimate_illuminant correction in [0,1] (default 0.6).",
    )


def log_priors(tag: str, model: DAINet, args) -> None:
    """Print which inference-time priors are active."""
    normals = "DSINE (computed inline)" if (
        getattr(model, "use_normals", False) and not getattr(args, "no_normals", False)
    ) else "none"
    # Inference illumination source: a predicted (φ,θ,b) direction (DirectionHead)
    # plus the latent illum_token when present, else just the learned null token.
    if getattr(model, "use_direction_head", False):
        illum = "predicted (φ,θ,b) + illum_token"
    elif getattr(model, "use_illum_token", False):
        illum = "illum_token (latent)"
    else:
        illum = "null token"
    if getattr(args, "estimate_illuminant", False):
        illum += f" + gray-world (strength={getattr(args, 'illuminant_strength', 0.6)})"
    print(f"[{tag}] priors: normals={normals}  illuminant={illum}", flush=True)


__all__ = [
    "load_image_rgb",
    "save_image_rgb",
    "load_model_from_checkpoint",
    "parse_size_arg",
    "infer_single",
    "resolve_device",
    "collect_inputs",
    "add_prior_args",
    "log_priors",
]
