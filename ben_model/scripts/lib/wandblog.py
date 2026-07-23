"""Lightweight wandb logger for the benchmark trainers.

Mirrors the *spirit* of dainet's ablation logging (one wandb run per model, live
training-loss + validation-metric curves, resumable) without dainet's rigid
7-section routing — the baselines have different (or single-term) losses, so we
keep two simple, numerically-prefixed sections that sort correctly in the UI:

    1_train/{loss,lr,sec_per_it}            per-step / per-epoch training scalars
    2_val/{l1,psnr,ssim,lpips}              probe-masked validation metrics

The x-axis is ``global_step`` (logged as a field, à la the dainet logger) so the
five models share one comparable iteration axis in the project.

Enabled by default (``mode="online"``, matching the ablation runs); pass
``mode="disabled"`` (``--no_wandb``) or set ``WANDB_MODE=disabled`` to make every
method a no-op. A failed ``wandb.init`` also degrades to a no-op so a wandb or
network problem never breaks training.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_PROJECT = "dainet-benchmark"


def read_run_id(out_dir: Path | str) -> str | None:
    """Recover a prior run id (written by a previous run) for resume."""
    sidecar = Path(out_dir) / "wandb_id.txt"
    if sidecar.exists():
        rid = sidecar.read_text().strip()
        return rid or None
    return None


class BenchWandb:
    def __init__(self, *, model: str, config: dict | None = None,
                 run_dir: Path | str | None = None, mode: str = "online",
                 project: str | None = None, entity: str | None = None,
                 run_name: str | None = None, tags: list[str] | None = None,
                 resume_id: str | None = None):
        # An explicit WANDB_MODE in the environment wins (global off-switch).
        mode = os.environ.get("WANDB_MODE") or mode
        self.enabled = mode != "disabled"
        self.run = None
        self.run_id = None
        self._wandb = None
        if not self.enabled:
            return
        try:
            import wandb

            self._wandb = wandb
            self.run = wandb.init(
                project=project or os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT),
                entity=entity or os.environ.get("WANDB_ENTITY"),
                name=run_name or f"{model}_mitmi",
                tags=tags or ["benchmark", model],
                config=config or {},
                mode=mode,
                dir=str(run_dir) if run_dir else None,
                id=resume_id,
                resume="allow" if resume_id else None,
                # Capture print()/tqdm into the run's "Logs" tab even when piped.
                settings=wandb.Settings(console="wrap"),
            )
            self.run_id = getattr(self.run, "id", None)
            if run_dir is not None and self.run_id:
                try:
                    (Path(run_dir) / "wandb_id.txt").write_text(str(self.run_id))
                except Exception:
                    pass
            wandb.define_metric("global_step")
            for prefix in ("1_train", "2_val"):
                wandb.define_metric(f"{prefix}/*", step_metric="global_step")
        except Exception as exc:  # never let wandb break training
            print(f"[wandb] disabled after init failure: {exc}", flush=True)
            self.enabled = False
            self.run = None
            self._wandb = None

    # ----------------------------------------------------------------- write
    def _emit(self, payload: dict[str, Any], step: int | None) -> None:
        if not self.enabled or self._wandb is None or not payload:
            return
        if step is not None:
            payload = {**payload, "global_step": step}
        # Do not pass step= so a resumed run keeps appending (wandb restores its
        # own internal counter); the x-axis uses the global_step field instead.
        self._wandb.log(payload)

    def log_train(self, scalars: dict[str, float], step: int | None = None) -> None:
        self._emit({f"1_train/{k}": v for k, v in scalars.items() if v is not None}, step)

    def log_val(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._emit({f"2_val/{k}": v for k, v in metrics.items() if v is not None}, step)

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
