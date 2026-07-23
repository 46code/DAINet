"""Direction-stratified failure analysis tooling.

Buckets per-image LPIPS errors into (φ, θ) bins and keeps a top-K worst-sample
buffer (with the input/target/prediction so the visualizer can draw a
self-describing comparison grid). A sample is flagged a failure when
``LPIPS > 0.25`` or ``PSNR < 18``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .metrics import lpips_score, psnr


@dataclass
class FailureBuckets:
    phi_bins: int = 8
    theta_bins: int = 4
    worst_top_k: int = 5
    sort_metric: str = "lpips"  # "lpips" (descending) or "psnr" (ascending)
    sums: np.ndarray = field(init=False)
    counts: np.ndarray = field(init=False)
    worst_samples: list[dict[str, Any]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self.sums = np.zeros((self.phi_bins, self.theta_bins), dtype=np.float64)
        self.counts = np.zeros((self.phi_bins, self.theta_bins), dtype=np.int64)
        self.worst_samples = []

    def _bin(self, phi_val: float, theta_val: float) -> tuple[int, int]:
        pi = int(np.clip(phi_val / (2 * math.pi) * self.phi_bins, 0, self.phi_bins - 1))
        ti = int(np.clip(theta_val / (math.pi / 2) * self.theta_bins, 0, self.theta_bins - 1))
        return pi, ti

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        phi: torch.Tensor,
        theta: torch.Tensor,
        scenes: list[str] | None = None,
        input_rgb: torch.Tensor | None = None,
    ) -> None:
        lpips_b = lpips_score(pred, target)  # [B]
        psnr_b = psnr(pred, target)  # [B]
        for b in range(pred.shape[0]):
            pi, ti = self._bin(float(phi[b]), float(theta[b]))
            lpips_val = float(lpips_b[b])
            psnr_val = float(psnr_b[b])
            self.sums[pi, ti] += lpips_val
            self.counts[pi, ti] += 1
            self.worst_samples.append(
                {
                    "lpips": lpips_val,
                    "psnr": psnr_val,
                    "is_failure": lpips_val > 0.25 or psnr_val < 18.0,
                    "phi": float(phi[b]),
                    "theta": float(theta[b]),
                    "scene": scenes[b] if scenes is not None else None,
                    "input": (input_rgb[b].detach().cpu() if input_rgb is not None else pred[b].detach().cpu()),
                    "pred": pred[b].detach().cpu(),
                    "target": target[b].detach().cpu(),
                }
            )
        if self.sort_metric == "psnr":
            self.worst_samples.sort(key=lambda e: e["psnr"])
        else:
            self.worst_samples.sort(key=lambda e: -e["lpips"])
        self.worst_samples = self.worst_samples[: self.worst_top_k]

    def heatmap(self) -> np.ndarray:
        counts_safe = np.maximum(self.counts, 1)
        return self.sums / counts_safe

    def save_heatmap_png(self, out_path: Path) -> Path:
        import matplotlib.pyplot as plt

        hm = self.heatmap()
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(hm.T, aspect="auto", cmap="hot", origin="lower")
        ax.set_xlabel(f"φ bin (0..{self.phi_bins - 1})")
        ax.set_ylabel(f"θ bin (0..{self.theta_bins - 1})")
        ax.set_title("Mean LPIPS by (φ, θ) bin")
        fig.colorbar(im, ax=ax)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        return out_path
