"""Precompute surface normals for the dainet encoder normals input.

Runs a monocular surface-normal estimator on each scene's first
directional view (``dir_0_mip2.jpg`` by default) and writes the
predicted normal map to::

    <normals_root>/<split>/<scene>/normal_mip2.npy

Cache is *soft-required*: scenes without a ``normal_mip2.npy`` emit
``has_normals = False`` from the dataloader and the encoder's
pre-fuse 1x1 conv treats that path as a zero-normals input (the
trained RGB-channel weights still apply).

Why one normal map per scene, not per direction?
------------------------------------------------
Surface normals are a geometric property of the scene that does not
depend on lighting direction. Estimators are robust enough to flash
shading that running once on ``dir_0`` is sufficient and 25× cheaper
than per-direction.

Engines
-------
``--engine marigold`` (default) — Marigold-Normals v1-1
(Ke et al., 2024) via HuggingFace ``diffusers``. Pip-installable
(``pip install -U diffusers transformers accelerate``). High-quality
diffusion-based normals; ~1-2 s/image on a single GPU at the default
4-step LCM schedule. Weights download automatically on first run from
``prs-eth/marigold-normals-v1-1`` (~3 GB cached under
``~/.cache/huggingface``).

``--engine dsine`` — DSINE (Bae & Davison, CVPR 2024). Faster than
Marigold (~50 ms / image on GPU). Uses the official repo via
``torch.hub.load(..., source="local")``; clone it once with::

    git clone https://github.com/baegwangbin/DSINE.git ~/my_model/DSINE
    curl -L -o ~/my_model/DSINE/checkpoints/dsine.pt \\
        https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt
    pip install geffnet

The script defaults ``DSINE_REPO`` to ``~/my_model/DSINE`` and
``DSINE_CKPT`` to ``<repo>/checkpoints/dsine.pt``; override either
via the matching env var if your clone lives elsewhere.

``--engine constant_up`` — fallback that writes the constant
``(0, 0, 1)`` normal for every pixel. Useful for smoke tests of the
training pipeline without any monocular-normals model installed; the
encoder's pre-fuse 1×1 conv learns to ignore a constant input. Not
for final runs.

Output format: ``float32`` ``[H, W, 3]`` in ``[-1, 1]`` (camera-space
right/up/out), saved via ``np.save``. The dataloader resizes and
transposes to ``[3, H, W]`` at sample time.

Usage
-----
    python -m scripts.precompute_normals \\
        --jpg_root data/raw/mit_mi/jpg \\
        --out      data/raw/mit_mi/normals \\
        --splits train val \\
        --engine marigold

The MIT-MI test split is intentionally excluded by default
(benchmark fairness — same policy as the SAM / superpixel caches).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------------------------- engines


def _run_constant_up(rgb_u8: np.ndarray) -> np.ndarray:
    """Constant (0, 0, 1) camera-space normal for every pixel."""
    h, w = rgb_u8.shape[:2]
    n = np.zeros((h, w, 3), dtype=np.float32)
    n[..., 2] = 1.0
    return n


def _load_marigold(
    model_id: str = "prs-eth/marigold-normals-v1-1",
    num_inference_steps: int = 4,
):
    """Load Marigold-Normals via diffusers and return a callable estimator.

    The returned callable accepts an HxWx3 uint8 RGB array and returns
    HxWx3 float32 normals in [-1, 1] at the input resolution.
    """
    try:
        import torch
        from diffusers import MarigoldNormalsPipeline
    except Exception as exc:
        raise RuntimeError(
            "diffusers / transformers / accelerate not available. Install with:\n"
            "  pip install -U diffusers transformers accelerate\n"
            "or pass --engine constant_up for a smoke-test fallback."
        ) from exc
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    pipe = MarigoldNormalsPipeline.from_pretrained(model_id, torch_dtype=dtype)
    pipe = pipe.to(device)

    @torch.inference_mode()
    def _estimate(rgb_u8: np.ndarray) -> np.ndarray:
        img = Image.fromarray(rgb_u8)
        out = pipe(image=img, num_inference_steps=num_inference_steps)
        # `MarigoldNormalsOutput.prediction` is a numpy array shaped
        # [B, H, W, 3] in [-1, 1] (verified against diffusers 0.38).
        pred = out.prediction
        if hasattr(pred, "detach"):  # in case a future diffusers returns a tensor
            pred = pred.detach().to(torch.float32).cpu().numpy()
        arr = np.asarray(pred)
        if arr.ndim == 4:
            arr = arr[0]
        return arr.astype(np.float32)

    return _estimate


def _load_dsine(device: str | None = None):
    """Load the official DSINE repo (CVPR 2024) via direct module import.

    ``device`` (e.g. ``"cuda:1"``) pins the estimator to a specific GPU; when
    None it uses the default cuda device (cuda:0) / CPU fallback. The precompute
    CLI masks GPUs via CUDA_VISIBLE_DEVICES and so leaves this None, but callers
    that select a logical cuda index (e.g. inference with ``--gpu N``) pass it
    through so DSINE follows the same device as the rest of the pipeline.

    Requires:
      - the DSINE repo cloned at ``DSINE_REPO`` (default ``~/my_model/DSINE``)
      - the dsine.pt checkpoint at ``DSINE_CKPT`` (default
        ``<repo>/checkpoints/dsine.pt``)
      - the ``geffnet`` PyPI package (``pip install geffnet``)

    The upstream ``hubconf.py`` is broken in the current DSINE repo
    (it does ``from models import dsine; dsine.DSINE()`` but the
    ``models/dsine/`` package has no top-level ``DSINE`` class — only
    ``v00`` / ``v01`` / ``v02`` / ``v02_kappa`` variants). So we
    replicate the canonical ``projects/dsine/test_minimal.py`` recipe
    directly: build an argparse Namespace matching the exp001 config,
    instantiate ``DSINE_v02(args)``, load the checkpoint, pad + run.

    DSINE's repo ships top-level ``models/`` and ``utils/`` packages
    that collide with this project's ``models/`` package. Work around
    it by synthesizing fake top-level ``models`` / ``utils`` modules
    whose ``__path__`` points at DSINE's directories so ``import
    models.dsine.v02`` and ``import utils.utils`` resolve to DSINE's
    files via ``sys.modules`` first, then restore ours afterwards.
    """
    import argparse
    import sys
    import types

    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:
        raise RuntimeError("PyTorch is required for the dsine engine.") from exc

    repo_path = Path(os.environ.get("DSINE_REPO", str(Path.home() / "my_model" / "DSINE"))).expanduser()
    if not (repo_path / "models" / "dsine" / "v02.py").exists():
        raise RuntimeError(
            f"DSINE repo not found at {repo_path}. Clone it with:\n"
            f"  git clone https://github.com/baegwangbin/DSINE.git {repo_path}\n"
            "or set DSINE_REPO to your clone path. "
            "Alternatively, pass --engine marigold (uses HuggingFace diffusers, "
            "no manual clone)."
        )
    ckpt_path = Path(os.environ.get(
        "DSINE_CKPT", str(repo_path / "checkpoints" / "dsine.pt")
    )).expanduser()
    if not ckpt_path.exists():
        raise RuntimeError(
            f"DSINE checkpoint not found at {ckpt_path}. Download with:\n"
            f"  curl -L -o {ckpt_path} https://huggingface.co/camenduru/DSINE/resolve/main/dsine.pt\n"
            "or set DSINE_CKPT to your checkpoint path."
        )

    try:
        import geffnet  # noqa: F401  — DSINE's only extra runtime dep
    except Exception as exc:
        raise RuntimeError(
            "DSINE requires `geffnet`. Install with: pip install geffnet"
        ) from exc

    # --- argparse Namespace: exp001_cvpr2024/dsine.txt defaults
    # (see projects/dsine/config.py for the parser; everything not set
    # below uses the parser's default value) ---
    args = argparse.Namespace(
        NNET_architecture="v02",
        NNET_output_dim=3,
        NNET_output_type="R",
        NNET_feature_dim=64,
        NNET_hidden_dim=64,
        NNET_encoder_B=5,
        NNET_decoder_NF=2048,
        NNET_decoder_BN=False,
        NNET_decoder_down=8,
        NNET_learned_upsampling=True,
        NRN_prop_ps=5,
        NRN_num_iter_train=5,
        NRN_num_iter_test=5,
        NRN_ray_relu=True,
    )

    # --- import-collision shim: install fake top-level `models` and
    # `utils` packages pointing at the DSINE clone so `import
    # models.dsine.v02` and `import utils.utils` resolve correctly.
    saved_models = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "models" or k.startswith("models.")
    }
    saved_utils = {
        k: sys.modules.pop(k)
        for k in list(sys.modules)
        if k == "utils" or k.startswith("utils.")
    }
    fake_models = types.ModuleType("models")
    fake_models.__path__ = [str(repo_path / "models")]  # type: ignore[attr-defined]
    sys.modules["models"] = fake_models
    fake_utils = types.ModuleType("utils")
    fake_utils.__path__ = [str(repo_path / "utils")]  # type: ignore[attr-defined]
    sys.modules["utils"] = fake_utils
    sys.path.insert(0, str(repo_path))
    try:
        from models.dsine.v02 import DSINE_v02  # type: ignore
        from utils.projection import intrins_from_fov  # type: ignore

        if device is not None:
            device = torch.device(device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DSINE_v02(args).to(device)
        state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.eval()
        # `pixel_coords` was allocated on cuda:0 inside __init__; if we are
        # using a non-default cuda index, move it to the model's device.
        model.pixel_coords = model.pixel_coords.to(device)
    finally:
        # Drop DSINE's `models.*` (the trained `model` keeps strong refs
        # to its classes so it stays usable) and restore ours.
        for k in list(sys.modules):
            if k == "models" or k.startswith("models."):
                del sys.modules[k]
        sys.modules.update(saved_models)
        # Keep DSINE's `utils.*` cached — there is no top-level `utils`
        # package in this project so no collision, and a future call into
        # any helper that does `import utils.xxx` keeps working.
        if saved_utils:
            for k in list(sys.modules):
                if k == "utils" or k.startswith("utils."):
                    del sys.modules[k]
            sys.modules.update(saved_utils)
        try:
            sys.path.remove(str(repo_path))
        except ValueError:
            pass

    # ImageNet normalization (matches test_minimal.py).
    _IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    _IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def _pad_to_multiple_of_32(orig_H: int, orig_W: int) -> tuple[int, int, int, int]:
        l = ((orig_W - 1) // 32 + 1) * 32 - orig_W
        t = ((orig_H - 1) // 32 + 1) * 32 - orig_H
        return l - l // 2, l // 2, t - t // 2, t // 2  # (left, right, top, bottom)

    @torch.inference_mode()
    def _estimate(rgb_u8: np.ndarray) -> np.ndarray:
        orig_H, orig_W = rgb_u8.shape[:2]
        x = (
            torch.from_numpy(rgb_u8.astype(np.float32) / 255.0)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device)
        )
        l, r, t, b = _pad_to_multiple_of_32(orig_H, orig_W)
        x = F.pad(x, (l, r, t, b), mode="constant", value=0.0)
        x = (x - _IMNET_MEAN) / _IMNET_STD
        # 60° FoV assumption — same default test_minimal.py uses when no
        # camera intrinsics .txt is supplied. MIT-MI doesn't ship per-image
        # intrinsics; the model is tolerant to approximate intrinsics per
        # the DSINE README.
        intrins = intrins_from_fov(new_fov=60.0, H=orig_H, W=orig_W, device=device).unsqueeze(0)
        intrins[:, 0, 2] += l
        intrins[:, 1, 2] += t
        pred = model(x, intrins=intrins)[-1]
        pred = pred[:, :, t:t + orig_H, l:l + orig_W]
        # pred is [1, 3, H, W] in [-1, 1]
        arr = pred[0].detach().to(torch.float32).cpu().numpy().transpose(1, 2, 0)
        return arr

    return _estimate


ENGINES = {
    "constant_up": _run_constant_up,
    "marigold": None,  # lazy-loaded via _load_marigold
    "dsine": None,     # lazy-loaded via _load_dsine
}


# --------------------------------------------------------------------- driver


def _enumerate_scenes(jpg_root: Path, split: str) -> list[Path]:
    sd = jpg_root / split
    if not sd.exists():
        return []
    return sorted(p for p in sd.iterdir() if p.is_dir())


def _load_view(scene_dir: Path, direction_id: int) -> np.ndarray | None:
    p = scene_dir / f"dir_{direction_id}_mip2.jpg"
    if not p.exists():
        return None
    bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jpg_root", default="data/raw/mit_mi/jpg")
    parser.add_argument("--out", default="data/raw/mit_mi/normals")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument(
        "--engine", choices=sorted(ENGINES.keys()), default="dsine",
        help="Normal-estimation engine. dsine (default) = official DSINE repo "
        "via torch.hub local (CVPR 2024, ~50 ms/img on GPU). marigold = "
        "Marigold-Normals v1-1 via diffusers (higher quality, ~1-2 s/img). "
        "constant_up = (0, 0, 1) everywhere (smoke test only).",
    )
    parser.add_argument(
        "--marigold_model", default="prs-eth/marigold-normals-v1-1",
        help="Marigold-Normals HuggingFace model id (used when --engine marigold).",
    )
    parser.add_argument(
        "--marigold_steps", type=int, default=4,
        help="Number of denoising steps for Marigold (1-50; 4 is a good speed/quality tradeoff).",
    )
    parser.add_argument(
        "--direction_id", type=int, default=0,
        help="Which directional view to estimate normals from (default 0).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-estimate even if the cache file already exists.",
    )
    parser.add_argument(
        "--gpu", type=int, default=None,
        help="CUDA device index to run the normal estimator on. Masks all "
        "other GPUs via CUDA_VISIBLE_DEVICES (the engine then uses cuda:0). "
        "Omit for the default device / CPU fallback.",
    )
    args = parser.parse_args(argv)

    # Mask to the requested GPU BEFORE torch initialises a CUDA context. The
    # engine loaders import torch lazily, so setting this here (pre-load) makes
    # the requested physical GPU the only visible device → logical cuda:0.
    if args.gpu is not None:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    jpg_root = Path(args.jpg_root)
    out_root = Path(args.out)

    if args.engine == "marigold":
        estimate = _load_marigold(
            model_id=args.marigold_model,
            num_inference_steps=args.marigold_steps,
        )
    elif args.engine == "dsine":
        estimate = _load_dsine()
    else:
        estimate = ENGINES[args.engine]

    n_done = 0
    n_skipped = 0
    n_failed = 0

    for split in args.splits:
        scenes = _enumerate_scenes(jpg_root, split)
        if not scenes:
            print(f"[normals] split {split!r}: no scenes under {jpg_root}")
            continue
        out_split = out_root / split
        out_split.mkdir(parents=True, exist_ok=True)
        for sd in scenes:
            out_path = out_split / sd.name / "normal_mip2.npy"
            if out_path.exists() and not args.overwrite:
                n_skipped += 1
                continue
            rgb = _load_view(sd, args.direction_id)
            if rgb is None:
                print(f"[normals] {split}/{sd.name}: dir_{args.direction_id}_mip2.jpg missing")
                n_failed += 1
                continue
            try:
                n = estimate(rgb)
            except Exception as exc:  # noqa: BLE001
                print(f"[normals] {split}/{sd.name}: {exc!r}")
                n_failed += 1
                continue
            n = np.clip(n.astype(np.float32), -1.0, 1.0)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, n)
            n_done += 1
            if n_done % 25 == 0:
                print(f"[normals] {n_done} scenes done...")

    print(
        f"[normals] DONE engine={args.engine} done={n_done} "
        f"skipped(cached)={n_skipped} failed={n_failed}"
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
