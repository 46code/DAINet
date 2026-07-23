"""The wandb routing must match the docs/wandb.md 7-section spec:

  1_media / 2_train_losses / 3_val_live_losses / 4_total_train / 5_total_val
  / 6_val_live_metrics / 7_val_ema_metrics

Train-total and val-total are SEPARATE one-curve sections (own panels); there
is NO val_ema loss section (EMA contributes eval metrics only).
"""

from __future__ import annotations

from training.wandb_logger import WandbLogger


def test_train_iter_routing():
    payload = {
        "loss_total": 1.0,
        "loss/recon_l1": 0.5,
        "diag/region_segments_active": 64.0,
        "lr": 1e-5,
        "grad_norm": 0.4,
        "epoch": 2,
    }
    out = WandbLogger._route_train_iter(payload)
    # epoch passes through as its own standalone panel; lr/grad_norm/diag dropped.
    assert out == {"4_total_train/loss": 1.0, "2_train_losses/recon_l1": 0.5, "epoch": 2}, out


def test_val_live_routing():
    out = WandbLogger._route_val(
        {"loss_total": 0.9, "loss/recon_l1": 0.4}, phase="val_live"
    )
    assert out == {"5_total_val/loss": 0.9, "3_val_live_losses/recon_l1": 0.4}, out


def test_val_ema_losses_dropped():
    # EMA contributes eval metrics only — no loss section in the spec.
    out = WandbLogger._route_val(
        {"loss_total": 0.9, "loss/recon_l1": 0.4}, phase="val_ema"
    )
    assert out == {}, out


def test_section_prefixes_are_the_seven_spec_sections():
    from training import wandb_logger as wl

    prefixes = {
        wl._SEC_MEDIA, wl._SEC_TRAIN_LOSSES, wl._SEC_VAL_LIVE_LOSSES,
        wl._SEC_TOTAL_TRAIN, wl._SEC_TOTAL_VAL,
        wl._SEC_VAL_LIVE_METRICS, wl._SEC_VAL_EMA_METRICS,
    }
    assert prefixes == {
        "1_media", "2_train_losses", "3_val_live_losses",
        "4_total_train", "5_total_val", "6_val_live_metrics", "7_val_ema_metrics",
    }
