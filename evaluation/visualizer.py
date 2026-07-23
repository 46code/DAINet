"""Qualitative grid renderers for wandb media.

Two distinct grids:

- `make_val_samples_grid` — fixed val indices, shows [Input | Target | Pred |
  R (reflectance)]. Lets you watch the *same* images improve across epochs.
- `make_val_failures_grid` — worst-K samples by LPIPS, shows
  [Input | Target | Pred] (LPIPS shown in the row caption). Tells you *where* the model is
  failing this epoch.

The previous gray "L / max(L)" panel is removed — it added no qualitative
signal (L is roughly uniform after a few hundred steps and reads as solid
gray in the gallery). Use the failure heatmap instead to see where the
prediction is off.

Every panel has a title (label) so the wandb media viewer is self-describing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _to_hwc_np(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()


def make_val_samples_grid(
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    R: torch.Tensor,
    out_path: Path,
    captions: list[str] | None = None,
    max_samples: int = 4,
) -> Path:
    """Composite figure: rows = samples, cols = [Input | Target | Pred | R]."""
    import matplotlib.pyplot as plt

    K = min(max_samples, input_rgb.shape[0])
    panel_names = ["Input", "Target", "Pred (I_out)", "Reflectance R"]
    tensors = [input_rgb, target, pred, R]
    fig, axes = plt.subplots(K, 4, figsize=(12, 3 * K))
    if K == 1:
        axes = axes[None, :]
    for k in range(K):
        cap = captions[k] if captions and k < len(captions) else f"sample {k}"
        for col, (name, t) in enumerate(zip(panel_names, tensors)):
            axes[k, col].imshow(_to_hwc_np(t[k]))
            title = name if k == 0 else ""
            if col == 0:
                axes[k, col].set_ylabel(cap, fontsize=8, rotation=0, ha="right", va="center")
            axes[k, col].set_title(title)
            axes[k, col].set_xticks([])
            axes[k, col].set_yticks([])
    fig.suptitle("val/samples — fixed-index val tracking", fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_qualitative_grid(
    input_rgb: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    out_path: Path,
    captions: list[str] | None = None,
    max_samples: int = 4,
) -> Path:
    """Main paper qualitative figure: rows = samples, cols = [Input | Pred | GT].

    Deliberately *no* error heatmap or reflectance panel — the LPIPS-ranked
    worst-K failure grid + (φ,θ) analysis live separately, and R/L live in the
    decomposition figure. This keeps the headline comparison clean."""
    import matplotlib.pyplot as plt

    K = min(max_samples, input_rgb.shape[0])
    panel_names = ["Input", "Prediction", "Ground-Truth"]
    tensors = [input_rgb, pred, target]
    fig, axes = plt.subplots(K, 3, figsize=(9, 3 * K))
    if K == 1:
        axes = axes[None, :]
    for k in range(K):
        cap = captions[k] if captions and k < len(captions) else f"sample {k}"
        for col, (name, t) in enumerate(zip(panel_names, tensors)):
            axes[k, col].imshow(_to_hwc_np(t[k]))
            if col == 0:
                axes[k, col].set_ylabel(cap, fontsize=13, fontweight="bold",
                                        rotation=0, ha="right", va="center")
            axes[k, col].set_title(name if k == 0 else "", fontsize=15, fontweight="bold")
            axes[k, col].set_xticks([])
            axes[k, col].set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_decomposition_grid(
    input_rgb: torch.Tensor,
    R: torch.Tensor,
    L: torch.Tensor,
    out_path: Path,
    S: torch.Tensor | None = None,
    captions: list[str] | None = None,
    max_samples: int = 4,
) -> Path:
    """Appendix intrinsic-decomposition figure: [Input | R | L | (Specular S)].

    Showcases the R·L factorisation. The specular column is included only when
    a specular map is supplied (the head is on)."""
    import matplotlib.pyplot as plt

    K = min(max_samples, input_rgb.shape[0])
    panel_names = ["Input", "Reflectance R", "Illumination L"]
    tensors = [input_rgb, R, L]
    if S is not None:
        panel_names.append("Specular S")
        tensors.append(S)
    ncol = len(tensors)
    fig, axes = plt.subplots(K, ncol, figsize=(3 * ncol, 3 * K))
    if K == 1:
        axes = axes[None, :]
    for k in range(K):
        cap = captions[k] if captions and k < len(captions) else f"sample {k}"
        for col, (name, t) in enumerate(zip(panel_names, tensors)):
            img = _to_hwc_np(t[k])
            axes[k, col].imshow(img if img.shape[-1] == 3 else img[..., 0], cmap=None if img.shape[-1] == 3 else "magma")
            if col == 0:
                axes[k, col].set_ylabel(cap, fontsize=13, fontweight="bold",
                                        rotation=0, ha="right", va="center")
            axes[k, col].set_title(name if k == 0 else "", fontsize=15, fontweight="bold")
            axes[k, col].set_xticks([])
            axes[k, col].set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_val_failures_grid(
    samples: list[dict],
    out_path: Path,
    sort_metric: str = "lpips",
) -> Path | None:
    """Composite figure of the worst-K val samples, ranked by `sort_metric`.

    Each row: [Input | Target | Pred], with the per-sample caption (scene,
    direction, and the ranking metric's value) as a row y-label. No in-figure
    title is drawn; the caller's caption states the ranking metric.

    Expects `samples` to be a list of dicts produced by FailureBuckets:
      {"input": [3,H,W], "pred": [3,H,W], "target": [3,H,W],
       "lpips": float, "psnr": float, "scene": str, "phi": float, "theta": float}
    """
    import matplotlib.pyplot as plt

    K = len(samples)
    if K == 0:
        return None
    fig, axes = plt.subplots(K, 3, figsize=(9, 3 * K))
    if K == 1:
        axes = axes[None, :]
    panel_names = ["Input", "Target", "Pred (I_out)"]
    metric_label = "PSNR" if sort_metric == "psnr" else "LPIPS"
    metric_fmt = "{:.2f}" if sort_metric == "psnr" else "{:.4f}"
    for k, s in enumerate(samples):
        row_caption = (
            f"{s.get('scene', '?')}\n"
            f"φ={s.get('phi', 0):.2f}\n"
            f"θ={s.get('theta', 0):.2f}\n"
            f"{metric_label}=" + metric_fmt.format(s.get(sort_metric, 0))
        )
        axes[k, 0].imshow(_to_hwc_np(s["input"]))
        axes[k, 0].set_ylabel(row_caption, fontsize=12, fontweight="bold",
                              rotation=0, ha="right", va="center")
        axes[k, 1].imshow(_to_hwc_np(s["target"]))
        axes[k, 2].imshow(_to_hwc_np(s["pred"]))
        for col, name in enumerate(panel_names):
            axes[k, col].set_title(name if k == 0 else "", fontsize=15, fontweight="bold")
            axes[k, col].set_xticks([])
            axes[k, col].set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_val_pair_images(
    input_rgb: torch.Tensor,
    pred: torch.Tensor,
    captions: list[str] | None = None,
) -> list:
    """Per-sample side-by-side `input | prediction` PIL images.

    Returns one PIL.Image per sample so the wandb media panel shows each
    pair as its own image (no grid). Caption (scene / direction) is
    rendered as a top strip on the composite.
    """
    from PIL import Image, ImageDraw, ImageFont

    out: list = []
    K = input_rgb.shape[0]
    for k in range(K):
        inp_np = (_to_hwc_np(input_rgb[k]) * 255.0).astype("uint8")
        pred_np = (_to_hwc_np(pred[k]) * 255.0).astype("uint8")
        h, w = inp_np.shape[:2]
        cap = (captions[k] if captions and k < len(captions) else f"sample {k}")
        cap_h = 22
        canvas = np.full((h + cap_h, 2 * w + 4, 3), 255, dtype=np.uint8)
        canvas[cap_h : cap_h + h, 0:w] = inp_np
        canvas[cap_h : cap_h + h, w + 4 : 2 * w + 4] = pred_np
        img = Image.fromarray(canvas)
        try:
            draw = ImageDraw.Draw(img)
            draw.text((4, 4), f"{cap}    |    input         predicted",
                      fill=(0, 0, 0))
        except Exception:
            pass
        out.append(img)
    return out


def make_val_failure_pair_images(
    samples: list[dict],
) -> list:
    """Per-sample side-by-side `input | prediction` PIL images for worst-K.

    Caption includes scene / phi / theta / LPIPS so the failures panel is
    self-describing without a heatmap grid.
    """
    from PIL import Image, ImageDraw

    out: list = []
    for s in samples:
        inp_np = (_to_hwc_np(s["input"]) * 255.0).astype("uint8")
        pred_np = (_to_hwc_np(s["pred"]) * 255.0).astype("uint8")
        h, w = inp_np.shape[:2]
        cap_h = 22
        canvas = np.full((h + cap_h, 2 * w + 4, 3), 255, dtype=np.uint8)
        canvas[cap_h : cap_h + h, 0:w] = inp_np
        canvas[cap_h : cap_h + h, w + 4 : 2 * w + 4] = pred_np
        img = Image.fromarray(canvas)
        cap = (
            f"{s.get('scene', '?')}  φ={s.get('phi', 0):.2f}  "
            f"θ={s.get('theta', 0):.2f}  LPIPS={s.get('lpips', 0):.4f}    "
            f"|    input         predicted"
        )
        try:
            draw = ImageDraw.Draw(img)
            draw.text((4, 4), cap, fill=(0, 0, 0))
        except Exception:
            pass
        out.append(img)
    return out


# Back-compat alias for older callers (smoke tests etc.).
def make_comparison_grid(
    input_rgb: torch.Tensor,
    target: torch.Tensor,
    pred: torch.Tensor,
    R: torch.Tensor,
    L: torch.Tensor,
    out_path: Path,
    max_samples: int = 4,
) -> Path:
    """Deprecated — kept for tests. Use make_val_samples_grid."""
    return make_val_samples_grid(input_rgb, target, pred, R, out_path, max_samples=max_samples)
