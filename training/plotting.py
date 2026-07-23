"""Loss-curve plotting utilities for saved training history."""

from __future__ import annotations

import json
from pathlib import Path


def plot_training_curves(history_path: Path, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    if not history_path.exists():
        return
    history = json.loads(history_path.read_text())
    if not history:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = [h["step"] for h in history if "loss_total" in h]
    losses = [h["loss_total"] for h in history if "loss_total" in h]
    if steps:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(steps, losses, label="total")
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title("Training loss")
        ax.legend()
        fig.savefig(out_dir / "loss_total.png", dpi=120, bbox_inches="tight")
        plt.close(fig)

    # Per-term losses
    term_keys = sorted({k for h in history for k in h.keys() if k.startswith("loss/")})
    if term_keys:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k in term_keys:
            xs = [h["step"] for h in history if k in h]
            ys = [h[k] for h in history if k in h]
            ax.plot(xs, ys, label=k.replace("loss/", ""))
        ax.set_xlabel("step")
        ax.set_ylabel("loss term")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title("Per-term loss")
        fig.savefig(out_dir / "loss_per_term.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
