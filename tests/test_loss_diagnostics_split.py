"""The loss manager must return (total, losses, diagnostics) with the total
aggregating weighted LOSSES ONLY — diagnostics (e.g. region_segments_active
≈ 64) must never leak into a loss total. This was the bug that made the val
loss_total ≈ 64 and broke the train-vs-val generalisation panel + the CSV.
"""

from __future__ import annotations

import torch

from losses.manager import DAINetLoss


def _fake_batch_and_out(B=2, H=24, W=32):
    out = {
        "output": torch.rand(B, 3, H, W),
        "reflectance": torch.rand(B, 3, H, W).clamp(1e-3, 1 - 1e-3),
        "illumination": torch.rand(B, 3, H, W) * 1.5 + 0.1,
        "sh_pred": torch.rand(B, 3, 25),
        "illuminant": torch.rand(B, 3) + 0.5,
    }
    batch = {
        "target": torch.rand(B, 3, H, W),
        "input_rgb": torch.rand(B, 3, H, W),
        "region_seg": torch.randint(0, 8, (B, 1, H, W)),
        "sh_target": torch.rand(B, 3, 25),
        "has_sh": torch.ones(B, dtype=torch.bool),
        "scene": ["s0"] * B,
    }
    return out, batch


def test_three_tuple_and_total_excludes_diagnostics():
    out, batch = _fake_batch_and_out()
    weights = {
        "recon_l1": 0.8, "log_chrom": 0.3, "delta_e2000": 0.3,
        "retinex_constraint": 0.02, "dir_consistency_R": 0.15, "probe_sh": 0.1,
        "xdir_relight": 0.1, "region_var": 0.1,
        "local_chroma": 0.05, "tv_L": 0.02, "illuminant_angular": 0.05,
    }
    lf = DAINetLoss(weights=weights, options={})
    lf.set_step(5000)
    ret = lf(out, batch)
    assert isinstance(ret, tuple) and len(ret) == 3, "must return 3-tuple"
    total, losses, diag = ret
    assert "region_segments_active" in diag
    assert "region_segments_active" not in losses
    recomputed = sum(float(v) for v in losses.values())
    assert abs(float(total) - recomputed) < 1e-4
    # The diagnostic value is large (~num segments); confirm it is NOT in total.
    assert float(total) < 10.0, "a ~64 diagnostic leaked into the loss total"


def test_new_terms_present_when_weighted():
    out, batch = _fake_batch_and_out()
    lf = DAINetLoss(
        weights={"recon_l1": 0.8, "local_chroma": 0.05, "tv_L": 0.02,
                 "illuminant_angular": 0.05},
        options={},
    )
    _, losses, _ = lf(out, batch)
    for k in ("local_chroma", "tv_L", "illuminant_angular"):
        assert k in losses and torch.isfinite(losses[k]).all()
