"""Wandb logger — 6-section layout pinned by numeric-prefixed metric keys.

The layout is the verbatim spec in ``docs/wandb.md`` (do exactly what it
says). Sections, in the order they appear in the run UI:

    1. 1_media/predictions/sample_*   — 3 prediction samples per step
       1_media/failures/sample_*      — 3 worst-failure samples per step
       1_media/other/*                — optional extra figures
    2. 2_train_losses/<term>          — per-term training losses (NO total)
    3. 3_val_live_losses/<term>       — per-term val-live losses (NO total)
    4. 4_total_train/loss             — total TRAIN loss (own panel)
    5. 5_total_val/loss               — total VAL loss (own panel)
    6. 6_val_live_metrics/<metric>    — live val eval metrics (PSNR, MS-SSIM, LPIPS)
    7. 7_val_ema_metrics/<metric>     — EMA  val eval metrics

Wandb groups panels by the longest common prefix; the numeric ``N_`` stem
forces the sections to sort in spec order in the sidebar. Train-total and
val-total now live in SEPARATE sections (4 / 5) so each renders as its own
panel; place them side by side to read the generalisation gap.

Spec notes (per ``docs/wandb.md``):
  - Train-total (``4_total_train/loss``) and val-total (``5_total_val/loss``)
    are SEPARATE one-curve sections, each its own panel.
  - NO ``val_ema_losses`` section — EMA contributes eval *metrics* only.
  - Eval metrics (PSNR, MS-SSIM, LPIPS) ARE now
    logged to wandb (sections 6 + 7). They previously lived in the CSV only.

System scalars (lr, grad_norm, timings) and the per-iter loss diagnostics
stay in ``logs/iter_history.jsonl``; the tabular metric history stays in
``logs/metrics_history.csv``. When ``wandb.mode == "disabled"`` every method
is a no-op.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


# Section prefixes (the leading integer fixes sidebar order).
_SEC_MEDIA = "1_media"
_SEC_TRAIN_LOSSES = "2_train_losses"
_SEC_VAL_LIVE_LOSSES = "3_val_live_losses"
# Train-total and val-total render as their OWN panels (separate sections) so
# each curve is read independently; the train↔val gap is still visible by
# placing the two panels side by side.
_SEC_TOTAL_TRAIN = "4_total_train"
_SEC_TOTAL_VAL = "5_total_val"
_SEC_VAL_LIVE_METRICS = "6_val_live_metrics"
_SEC_VAL_EMA_METRICS = "7_val_ema_metrics"


class WandbLogger:
    def __init__(self, cfg: dict, run_dir: Path | None = None, *, phase: str = "train"):
        # dainet policy: test never logs to wandb.
        if phase == "test":
            raise RuntimeError(
                "WandbLogger may not be instantiated in test phase "
                "(dainet policy: test never logs to wandb)."
            )
        wb_cfg = cfg.get("wandb", {})
        mode = wb_cfg.get("mode", "online")
        project = wb_cfg.get("project", "dainet")
        entity = wb_cfg.get("entity", None)
        run_name = wb_cfg.get("run_name", None)
        tags = wb_cfg.get("tags", None)
        # Resume: when a run id is supplied (from the run launcher reading the
        # checkpoint / wandb_id.txt), continue that same wandb run so metrics
        # append onto one curve instead of orphaning the prior run.
        resume_id = wb_cfg.get("id", None)
        self.enabled = mode != "disabled"
        self.run = None
        self.run_id = None
        self._wandb = None
        if self.enabled:
            import wandb

            self._wandb = wandb
            try:
                self.run = wandb.init(
                    project=project,
                    entity=entity,
                    name=run_name,
                    tags=tags,
                    config=cfg,
                    mode=mode,
                    dir=str(run_dir) if run_dir else None,
                    id=resume_id,
                    resume="allow" if resume_id else None,
                    # Capture stdout/stderr into the run's "Logs" tab. "wrap"
                    # wraps the Python stream objects so it still grabs print()
                    # and tqdm.write() even when the process output is piped
                    # (tmux / tee / nohup), unlike FD-level "redirect".
                    settings=wandb.Settings(console="wrap"),
                )
                self.run_id = getattr(self.run, "id", None)
                # Persist the id so a later --resume can recover this run even
                # if the checkpoint predates wandb-id tracking.
                if run_dir is not None and self.run_id:
                    try:
                        (Path(run_dir) / "wandb_id.txt").write_text(str(self.run_id))
                    except Exception:
                        pass
                self._define_sections()
            except Exception as exc:
                self.enabled = False
                self.run = None
                self._wandb = None
                print(f"[wandb] disabled after init failure: {exc}")

    # ------------------------------------------------------------------ setup
    def _define_sections(self) -> None:
        """Pin the 6 sections so the run UI is stable + correctly ordered."""
        wb = self._wandb
        if wb is None:
            return
        wb.define_metric("global_step")
        wb.define_metric("epoch", step_metric="global_step")
        for prefix in (
            _SEC_MEDIA,
            _SEC_TRAIN_LOSSES,
            _SEC_VAL_LIVE_LOSSES,
            _SEC_TOTAL_TRAIN,
            _SEC_TOTAL_VAL,
            _SEC_VAL_LIVE_METRICS,
            _SEC_VAL_EMA_METRICS,
        ):
            wb.define_metric(f"{prefix}/*", step_metric="global_step")

    # ----------------------------------------------------------- key routing
    @staticmethod
    def _route_train_iter(payload: dict[str, Any]) -> dict[str, Any]:
        """Route a flat per-iter train payload into sections 2 and 4.

            "loss_total"  → "4_total_train/loss"
            "loss/<term>" → "2_train_losses/<term>"
            "epoch"       → "epoch" (standalone readout, outside the 6 sections)
            "diag/*", "lr", "grad_norm" → dropped (disk-side only)
        """
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if k == "loss_total":
                out[f"{_SEC_TOTAL_TRAIN}/loss"] = v
            elif k.startswith("loss/"):
                term = k.split("/", 1)[1]
                out[f"{_SEC_TRAIN_LOSSES}/{term}"] = v
            elif k == "epoch":
                # Standalone top-level metric, not part of any N_ section, so
                # it renders as its own panel without disturbing the spec.
                out["epoch"] = v
            # diag/*, lr, grad_norm and anything else is dropped:
            # the spec sections are exclusive.
        return out

    @staticmethod
    def _route_val(payload: dict[str, Any], *, phase: str) -> dict[str, Any]:
        """Route a val-loss payload.

        phase = "val_live":
            "loss_total"  → "5_total_val/loss"
            "loss/<term>" → "3_val_live_losses/<term>"

        phase = "val_ema": no loss section exists in the spec — drop
            everything (EMA contributes eval metrics only, via log_metrics).
        """
        if phase != "val_live":
            return {}
        out: dict[str, Any] = {}
        for k, v in payload.items():
            if k == "loss_total":
                out[f"{_SEC_TOTAL_VAL}/loss"] = v
            elif k.startswith("loss/"):
                term = k.split("/", 1)[1]
                out[f"{_SEC_VAL_LIVE_LOSSES}/{term}"] = v
        return out

    # ----------------------------------------------------------------- write
    def _emit(self, payload: dict[str, Any], step: int | None) -> None:
        """Send one row to wandb.

        ``global_step`` is logged as a FIELD (it is the ``step_metric`` for
        every panel — see ``_define_sections``) and we deliberately do NOT pass
        ``step=`` to ``wandb.log``. On a *resumed* run wandb restores its
        internal step counter to the run's last committed value; a
        ``log(step=N)`` call with ``N`` ≤ that restored value is dropped as
        "non-monotonic" ("Tried to log to step X that is less than the current
        step Y … this data will be ignored"), silently losing every post-resume
        point. Letting wandb auto-increment its own internal step keeps all
        resumed logs, while the charts still use ``global_step`` for the x-axis.
        """
        if not self.enabled or self._wandb is None:
            return
        if step is not None:
            payload = {**payload, "global_step": step}
        self._wandb.log(payload)

    def log(
        self,
        payload: dict[str, Any],
        step: int | None = None,
        *,
        phase: str | None = None,
    ) -> None:
        """Log a loss payload with phase-based key routing.

        - phase=None: pass through (caller pre-keyed).
        - phase="train": re-key per `_route_train_iter`.
        - phase="val_live": re-key per `_route_val`.
        - phase="val_ema": no-op (EMA losses are not a wandb section).
        """
        if not self.enabled or self._wandb is None:
            return
        if phase == "train":
            payload = self._route_train_iter(payload)
        elif phase in ("val_live", "val_ema"):
            payload = self._route_val(payload, phase=phase)
        if not payload:
            return
        self._emit(payload, step)

    def log_metrics(
        self,
        metrics: dict[str, float],
        *,
        phase: str,
        step: int | None = None,
    ) -> None:
        """Log the three benchmark eval metrics into section 6 (val_live) or
        section 7 (val_ema). `phase` must be "val_live" or "val_ema".
        """
        if not self.enabled or self._wandb is None:
            return
        if phase == "val_live":
            prefix = _SEC_VAL_LIVE_METRICS
        elif phase == "val_ema":
            prefix = _SEC_VAL_EMA_METRICS
        else:
            return
        payload: dict[str, Any] = {
            f"{prefix}/{k}": v for k, v in metrics.items()
        }
        if not payload:
            return
        self._emit(payload, step)

    def log_images(
        self,
        name: str,
        images: dict[str, Path | "PIL.Image.Image" | torch.Tensor],
        step: int | None = None,
    ) -> None:
        """Log images under ``1_media/<name>/<key>`` (section 1)."""
        if not self.enabled or self._wandb is None:
            return
        payload: dict[str, Any] = {}
        for k, v in images.items():
            log_key = f"{_SEC_MEDIA}/{name}/{k}"
            if isinstance(v, Path):
                payload[log_key] = self._wandb.Image(str(v))
            elif isinstance(v, torch.Tensor):
                arr = v.detach().cpu().clamp(0, 1)
                if arr.dim() == 3 and arr.shape[0] in (1, 3):
                    arr = arr.permute(1, 2, 0)
                payload[log_key] = self._wandb.Image(arr.numpy())
            else:
                payload[log_key] = self._wandb.Image(v)
        self._emit(payload, step)

    def log_gradients(self, model: torch.nn.Module, step: int | None = None) -> None:
        """No-op kept for back-compat. Grad norm lives in iter_history.jsonl."""
        if not self.enabled or self._wandb is None:
            return

    def finish(self) -> None:
        if self.enabled and self._wandb is not None:
            self._wandb.finish()
