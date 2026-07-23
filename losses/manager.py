"""DAINetLoss — YAML-driven composition of the dainet loss stack.

Active terms in the `dainet_full` baseline (see `configs/dainet.yaml`):

    recon_l1            0.80   sRGB pixel anchor
    log_chrom           0.30   linear-RGB log chromaticity invariance
    delta_e2000         0.30   Sharma 2005 ΔE2000 — headline perceptual color
    lpips               0.05   LPIPS-AlexNet (matches metric)
    dir_consistency_R   0.15   same-scene reflectance invariance
    probe_sh            0.10   chrome-probe SH supervision (train-only)
    xdir_relight        0.10*† cross-direction R · L (novelty B)
    retinex_constraint  0.02*† ‖I_out − R·L‖₁ — R·L = I anchor
    region_var          0.10   within GT-chroma-cluster chromaticity uniformity
    material_ce         0.05   MaterialHead per-pixel CE (aux supervision)
    material_R_var      0.10   within-material R variance (aux)
    local_chroma        0.05   edge-aware chroma-TV on I_out (cast removal)
    tv_L                0.02   edge-aware TV on L (smoothness regulariser)
    illuminant_angular  0.05   global illuminant angular supervision (aux)
    specular_bce        0.10   specular-head BCE (keeps highlights out of R)

Terms marked `*` are subject to a step-based linear warmup envelope.
Terms marked `†` are additionally held at zero for the first
`loss_factorisation_skip_iters` optimiser steps (default 100).

The disabled (weight-0) ablation slots that used to live here were removed
2026-06-05 — `pseudo_R_l1`, `illum_neutrality`, `lab_chroma_ab`, `dists`,
`region_mean`, `highlight`, `lab_consistency`, `multiscale_chroma`,
`edge_chroma`, `freq_decomp`, `fft_l1` — none of them were used by any run.

Probe-region and clipped-highlight pixels are masked out of every
reconstruction-style loss (`recon_l1`, `log_chrom`, `delta_e2000`,
`retinex_constraint`). Structural / region / direction-consistency /
identifiability terms are unaffected.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .directional import directional_R_consistency
from .log_chrom import log_chromaticity_loss
from .material import material_ce_loss
from .perceptual_color import DeltaE2000Loss
from .probe_sh import probe_sh_loss
from .reconstruction import recon_l1_loss
from .region import region_chroma_variance
from .retinex import RetinexConstraintLoss
from .xdir_relight import xdir_relighting_loss


def _expand_to(mask: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Broadcast/upcast a [B,1,H,W] or [1,H,W] mask to match ref's dtype/device."""
    m = mask
    if m.dim() == 3:
        m = m.unsqueeze(0)
    return m.to(device=ref.device, dtype=ref.dtype)


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return (pred - target).abs().mean()
    m = _expand_to(mask, pred).expand_as(pred)
    denom = m.sum().clamp_min(1.0)
    return ((pred - target).abs() * m).sum() / denom


def _build_highlight_mask(
    input_rgb: torch.Tensor, target: torch.Tensor, threshold: float
) -> torch.Tensor:
    """1 where neither input nor target is near-saturated; 0 where it is."""
    nz_target = (target < threshold).all(dim=1, keepdim=True)
    nz_input = (input_rgb < threshold).all(dim=1, keepdim=True)
    return (nz_target & nz_input).to(target.dtype)


# Sqrt floor for log_chrom inputs. 1e-4 is safe under bf16.
_LOG_EPS = 1e-4

# Default soft-warmup envelope: applied to the factorisation-coupling
# identifiability terms only. Pure pixel anchors, SH supervision, and
# the region/material priors do not warm up — they are always on so the
# network has a stable supervised target from step 0.
_DEFAULT_WARMUP_TERMS: frozenset[str] = frozenset(
    {
        "retinex_constraint",
        "xdir_relight",
    }
)
_DEFAULT_WARMUP_ITERS: int = 3000
_DEFAULT_WARMUP_FLOOR: float = 0.05
# Terms held entirely at zero until the network has any factorisation
# structure (R and L start identity-at-init). Empirically the bf16 noise
# floor in the first ~100 steps swung these by 5–6 orders of magnitude.
_DEFAULT_FACTORISATION_SKIP_ITERS: int = 100
_FACTORISATION_TERMS: frozenset[str] = frozenset(
    {"retinex_constraint", "xdir_relight"}
)


class DAINetLoss(nn.Module):
    def __init__(self, weights: dict[str, float], options: dict | None = None):
        super().__init__()
        self.weights: dict[str, float] = {k: float(v) for k, v in weights.items()}
        self.options = dict(options or {})

        # Step-based linear warmup envelope for "soft" terms. Effective weight
        # = base * max(floor, min(1, step / warmup_iters)). Floor keeps soft
        # terms contributing a small, fixed gradient from step 0 instead of
        # being amplified out of bf16 noise. The trainer calls `set_step`
        # once per optimizer step.
        self._warmup_iters: int = int(
            self.options.get("loss_warmup_iters", _DEFAULT_WARMUP_ITERS)
        )
        self._warmup_floor: float = float(
            self.options.get("loss_warmup_floor", _DEFAULT_WARMUP_FLOOR)
        )
        warmup_terms = self.options.get("loss_warmup_terms")
        if warmup_terms is None:
            self._warmup_terms = set(_DEFAULT_WARMUP_TERMS)
        else:
            self._warmup_terms = {str(t) for t in warmup_terms}
        self._factorisation_skip_iters: int = int(
            self.options.get(
                "loss_factorisation_skip_iters", _DEFAULT_FACTORISATION_SKIP_ITERS
            )
        )
        self._step: int = 0

        # Back-compat epoch-based schedule (used by some ablation configs).
        # If a term appears in both _schedules and _warmup_terms, the step
        # envelope is applied first, then the epoch interpolation on top —
        # but in practice we don't use both for the same term.
        self._schedules: dict[str, dict] = dict(self.options.get("loss_schedule", {}))
        self._epoch: int = 0

        # Stateful loss modules (kept around so their buffers move with .to()).
        self.delta_e2000_loss = DeltaE2000Loss()
        self.retinex_loss = RetinexConstraintLoss()

        # Lazy slots — modules built on first use so a config that zeroes a
        # term never pays its construction cost.
        self._specular_loss = None
        self._lpips_loss = None
        self._material_r_var_loss = None
        self._edge_tv = None
        self._local_chroma_loss = None
        self._illuminant_loss = None

    def set_epoch(self, epoch: int) -> None:
        """Tell the loss manager the current training epoch so any epoch-
        based scheduled weights can interpolate. Trainer calls this once
        per epoch."""
        self._epoch = int(epoch)

    def set_step(self, step: int) -> None:
        """Tell the loss manager the current global optimizer step so the
        step-based warmup envelope can ramp. Trainer calls this once per
        accumulated optimizer step (not per micro-batch)."""
        self._step = int(step)

    def _w(self, key: str) -> float:
        base = float(self.weights.get(key, 0.0))
        if base <= 0.0:
            return 0.0
        # Factorisation skip — hold retinex / xdir_relight at zero until R
        # and L have any structure (random at init).
        if (
            key in _FACTORISATION_TERMS
            and self._step < self._factorisation_skip_iters
        ):
            return 0.0
        # Step-based linear warmup envelope with floor.
        if key in self._warmup_terms and self._warmup_iters > 0:
            ramp = min(1.0, float(self._step) / float(self._warmup_iters))
            envelope = max(self._warmup_floor, ramp)
            base = base * envelope
        # Optional epoch-based schedule (back-compat). Linearly interpolates
        # from `start` at epoch 0 to `end` at `warmup_epochs`, then holds at
        # `end`. Stacks on top of the step envelope.
        sched = self._schedules.get(key)
        if sched:
            start = float(sched.get("start", 0.0))
            end = float(sched.get("end", base))
            warmup_e = max(int(sched.get("warmup_epochs", 0)), 0)
            if warmup_e <= 0 or self._epoch >= warmup_e:
                base = end
            else:
                frac = self._epoch / float(warmup_e)
                base = start + (end - start) * frac
        return base

    def effective_weight(self, key: str) -> float:
        """Public accessor for the current scheduled weight of a term —
        used by the trainer to log warmup-envelope values to wandb."""
        return self._w(key)

    def forward(
        self, out: dict[str, torch.Tensor], batch: dict
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        I_out = out["output"]
        R = out["reflectance"]
        L = out["illumination"]
        sh_pred = out["sh_pred"]

        target = batch["target"]
        input_rgb = batch["input_rgb"]
        # Dual segmentation (region losses).
        region_segments = batch.get("region_seg", batch.get("input_seg"))
        sh_target = batch["sh_target"]
        has_sh = batch["has_sh"]
        scenes = batch.get("scene", [])

        # ----- mask construction -----
        probe_mask = batch.get("probe_mask")
        if probe_mask is not None:
            probe_mask = _expand_to(probe_mask, target)

        highlight_threshold = float(self.options.get("highlight_mask_threshold", 0.98))
        hi_mask = _build_highlight_mask(input_rgb, target, highlight_threshold)
        recon_mask = hi_mask if probe_mask is None else (hi_mask * probe_mask)

        losses: dict[str, torch.Tensor] = {}
        diagnostics: dict[str, torch.Tensor] = {}

        # ---- recon ----
        if self._w("recon_l1") > 0:
            losses["recon_l1"] = self._w("recon_l1") * _masked_l1(I_out, target, recon_mask)
        if self._w("log_chrom") > 0:
            if recon_mask is None:
                losses["log_chrom"] = self._w("log_chrom") * log_chromaticity_loss(I_out, target)
            else:
                from .color_ops import srgb_to_linear

                p_lin = srgb_to_linear(I_out.clamp(0.0, 1.0)).float()
                t_lin = srgb_to_linear(target.clamp(0.0, 1.0)).float()
                diff = (
                    torch.log(p_lin.clamp_min(_LOG_EPS))
                    - torch.log(t_lin.clamp_min(_LOG_EPS))
                ).abs()
                m = recon_mask.to(diff.dtype).expand_as(diff)
                losses["log_chrom"] = self._w("log_chrom") * (
                    (diff * m).sum() / m.sum().clamp_min(1.0)
                )

        # ---- perceptual / chroma ----
        if self._w("delta_e2000") > 0:
            losses["delta_e2000"] = self._w("delta_e2000") * self.delta_e2000_loss(
                I_out, target, mask=recon_mask
            )
        if self._w("lpips") > 0:
            if self._lpips_loss is None:
                from .lpips_utils import lpips_loss

                self._lpips_loss = lpips_loss
            # AlexNet backbone — literature standard for low-level vision
            # benchmarks (Zhang CVPR 2018). The metric side uses the same
            # network so train-time supervision and val/test scoring are
            # measured against identical features.
            losses["lpips"] = self._w("lpips") * self._lpips_loss(
                I_out, target, net="alex"
            )

        # ---- identifiability / factorization ----
        if self._w("retinex_constraint") > 0:
            losses["retinex_constraint"] = self._w("retinex_constraint") * self.retinex_loss(
                I_out, R, L, mask=recon_mask
            )
        if (
            self._w("dir_consistency_R") > 0
            and isinstance(scenes, (list, tuple))
            and len(scenes) > 1
        ):
            losses["dir_consistency_R"] = self._w("dir_consistency_R") * directional_R_consistency(
                R, list(scenes)
            )
        if self._w("probe_sh") > 0:
            losses["probe_sh"] = self._w("probe_sh") * probe_sh_loss(sh_pred, sh_target, has_sh)
        if (
            self._w("xdir_relight") > 0
            and isinstance(scenes, (list, tuple))
            and len(scenes) > 1
        ):
            losses["xdir_relight"] = self._w("xdir_relight") * xdir_relighting_loss(
                R, L, input_rgb, list(scenes)
            )

        # ---- illumination smoothness (ablation / robust-config only) ----
        # Edge-aware TV on L: smooths illumination in flat input regions,
        # allows it to vary at input edges. Biases edges into R, smooth
        # fall-off into L. Off in the lean baseline (fights legitimate L
        # variation if over-weighted); on in the `robust` experiment.
        if self._w("tv_L") > 0:
            if self._edge_tv is None:
                from .tv import edge_aware_tv

                self._edge_tv = edge_aware_tv
            losses["tv_L"] = self._w("tv_L") * self._edge_tv(L, input_rgb)

        # ---- local-chroma cast removal (ablation / robust-config only) ----
        # Edge-aware chromaticity TV on I_out: penalises chroma gradients
        # only where the *input* has no luminance edge, so a smooth local
        # color cast (e.g. a red sidelight tinting one wall) is suppressed
        # while genuine material/chroma boundaries are preserved. Operates
        # on I_out (not R over SAM ids) so it cannot fight reconstruction —
        # see docs/dainet_losses.md "Where local chroma comes from now".
        if self._w("local_chroma") > 0:
            if self._local_chroma_loss is None:
                from .local_chroma import local_chroma_tv

                self._local_chroma_loss = local_chroma_tv
            losses["local_chroma"] = self._w("local_chroma") * self._local_chroma_loss(
                I_out, input_rgb
            )

        # ---- global illuminant angular supervision (training-only) ----
        # Supervises the optional IlluminantHead (out["illuminant"]) toward
        # the scene's true illuminant chromaticity (input/target mean ratio)
        # in angular space (training-only color-constancy supervision).
        if self._w("illuminant_angular") > 0 and "illuminant" in out:
            if self._illuminant_loss is None:
                from .illuminant import illuminant_angular_loss

                self._illuminant_loss = illuminant_angular_loss
            losses["illuminant_angular"] = self._w("illuminant_angular") * self._illuminant_loss(
                out["illuminant"], input_rgb, target, mask=recon_mask
            )

        # ---- light-direction prediction (DirectionHead, training-only) ----
        # Supervises the predicted (φ,θ,b) toward the GT capture metadata so the
        # head can replace the null token at inference. Gated by has_meta.
        if (
            self._w("direction_pred") > 0
            and "dir_pred_enc" in out
            and "phi" in batch
            and "brightness_norm" in batch
        ):
            from .direction import direction_pred_loss

            losses["direction_pred"] = self._w("direction_pred") * direction_pred_loss(
                out["dir_pred_enc"],
                batch["phi"],
                batch["theta"],
                batch["brightness_norm"],
                batch.get("has_meta"),
            )

        # ---- Region / material priors and other ablation slots ----
        if self._w("region_var") > 0:
            losses["region_var"] = self._w("region_var") * region_chroma_variance(
                I_out, region_segments
            )
            from .region import region_active_segment_count

            n_active = region_active_segment_count(region_segments)
            diagnostics["region_segments_active"] = torch.tensor(
                float(n_active), device=I_out.device, dtype=I_out.dtype
            )
        if (
            self._w("material_ce") > 0
            and "material_logits" in out
            and "material_seg" in batch
            and "has_material" in batch
        ):
            losses["material_ce"] = self._w("material_ce") * material_ce_loss(
                out["material_logits"],
                batch["material_seg"],
                probe_mask=probe_mask,
                has_material=batch["has_material"],
            )
        if self._w("specular_bce") > 0 and "specular_logit" in out:
            if self._specular_loss is None:
                from .specular import SpecularBCELoss

                self._specular_loss = SpecularBCELoss().to(I_out.device)
            losses["specular_bce"] = self._w("specular_bce") * self._specular_loss(
                out["specular_logit"], input_rgb
            )
        if (
            self._w("material_R_var") > 0
            and "material_seg" in batch
            and "has_material" in batch
        ):
            num_classes = int(self.options.get("material_num_classes", 0))
            if num_classes <= 0 and "material_logits" in out:
                num_classes = int(out["material_logits"].shape[1])
            if num_classes > 0:
                if self._material_r_var_loss is None:
                    from .material import material_R_variance_loss

                    self._material_r_var_loss = material_R_variance_loss
                losses["material_R_var"] = self._w("material_R_var") * self._material_r_var_loss(
                    R,
                    batch["material_seg"],
                    has_material=batch["has_material"],
                    num_classes=num_classes,
                )

        if losses:
            total = torch.stack(list(losses.values())).sum()
        else:
            total = torch.zeros((), device=I_out.device, dtype=I_out.dtype)
        # Return losses and diagnostics SEPARATELY. `total` and every
        # downstream `*_total` sum must aggregate weighted *losses only*;
        # diagnostics (e.g. `region_segments_active` ≈ 64, `dists_alpha_*`)
        # are instrumentation and must never enter a loss total. (Folding
        # them in previously made the val loss_total ≈ 64 and broke the
        # train-vs-val generalisation panel + metrics_history.csv.)
        return total, losses, diagnostics
