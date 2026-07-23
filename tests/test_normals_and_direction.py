"""Invariants for the 2026-06-03 architecture levers:

- ``normals_fusion`` in {none, prefuse, encoder} all preserve identity-at-init
  (``I_out == input_rgb`` at step 0), and the ``encoder`` path adds a
  zero residual at init so it matches RGB-only.
- categorical direction encoding builds and runs, and the zero-init embedding
  keeps identity-at-init (consumers are zero-init regardless).
- ``activation_checkpoint`` is numerically a no-op in eval / no-grad.
"""

from __future__ import annotations

import torch

from models.network import DAINet


def _build(**kw):
    return DAINet(
        embed_dim=64, seg_embed_dim=64, illum_hidden=64, attn_heads=4,
        pretrained_encoder=False, backbone="convnext_tiny",
        use_swin_bottleneck=True, use_illum_token=True,
        **kw,
    )


def test_identity_at_init_all_normals_modes():
    rgb = torch.rand(2, 3, 64, 80)
    normals = torch.rand(2, 3, 64, 80) * 2 - 1
    for mode in ("none", "prefuse", "encoder"):
        torch.manual_seed(0)
        m = _build(normals_fusion=mode).eval()
        with torch.no_grad():
            out = m(rgb, normals=None if mode == "none" else normals)
        assert (out["output"] - rgb).abs().max().item() < 1e-3, f"identity broken: {mode}"


def test_encoder_fusion_residual_zero_at_init():
    # encoder mode at init must equal a none-mode forward (zero-init projections).
    rgb = torch.rand(2, 3, 64, 80)
    normals = torch.rand(2, 3, 64, 80) * 2 - 1
    torch.manual_seed(0)
    m_none = _build(normals_fusion="none").eval()
    torch.manual_seed(0)
    m_enc = _build(normals_fusion="encoder").eval()
    # Share all overlapping weights; only the (zero-init) normals encoder differs.
    m_enc.load_state_dict(m_none.state_dict(), strict=False)
    with torch.no_grad():
        o_none = m_none(rgb)["output"]
        o_enc = m_enc(rgb, normals=normals)["output"]
    assert (o_enc - o_none).abs().max().item() < 1e-5


def test_use_normals_legacy_maps_to_prefuse():
    m = _build(use_normals=True)
    assert m.normals_fusion == "prefuse"
    m2 = _build(use_normals=False)
    assert m2.normals_fusion == "none"


def test_categorical_direction_encoding():
    torch.manual_seed(0)
    m = _build(direction_encoding="categorical", num_directions=25).eval()
    rgb = torch.rand(2, 3, 48, 48)
    did = torch.tensor([3, 17], dtype=torch.long)
    with torch.no_grad():
        out = m(rgb, direction_id=did)
    # zero-init embedding ⇒ identity-at-init still holds
    assert (out["output"] - rgb).abs().max().item() < 1e-3
    # null fallback (no direction_id) also runs
    with torch.no_grad():
        out2 = m(rgb)
    assert out2["output"].shape == rgb.shape


def test_activation_checkpoint_noop_in_eval():
    torch.manual_seed(0)
    m = _build(normals_fusion="encoder", activation_checkpoint=True).eval()
    rgb = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = m(rgb)
    # eval + no-grad ⇒ checkpointing path is skipped, forward still valid.
    assert (out["output"] - rgb).abs().max().item() < 1e-3


# ----------------------- light-direction head (contribution C) -----------------------

def test_direction_head_runs_and_identity_at_init():
    torch.manual_seed(0)
    m = _build(use_direction_head=True, direction_encoding="continuous").eval()
    assert m.use_direction_head is True
    rgb = torch.rand(2, 3, 64, 64)
    with torch.no_grad():
        out = m(rgb)  # no (φ,θ,b) ⇒ inference path uses the PREDICTED direction
    # head emits a 5-dim encoding with unit (sin, cos) pairs
    enc = out["dir_pred_enc"]
    assert enc.shape == (2, 5)
    assert torch.allclose(enc[:, 0:2].norm(dim=-1), torch.ones(2), atol=1e-4)
    assert torch.allclose(enc[:, 2:4].norm(dim=-1), torch.ones(2), atol=1e-4)
    # identity-at-init holds even though the predicted illum_emb is non-zero
    # (the downstream fusion / FiLM / heads are all zero-init).
    assert (out["output"] - rgb).abs().max().item() < 1e-3


def test_direction_head_teacher_forcing_and_pred_paths():
    torch.manual_seed(0)
    m = _build(use_direction_head=True).eval()
    rgb = torch.rand(2, 3, 48, 48)
    phi, theta, b = torch.rand(2), torch.rand(2), torch.rand(2) + 0.5
    with torch.no_grad():
        o_tf = m(rgb, phi=phi, theta=theta, bnorm=b)                         # teacher forcing (GT)
        o_pred = m(rgb, phi=phi, theta=theta, bnorm=b, use_pred_direction=True)  # predicted
    # the head runs (and exposes its prediction) in BOTH cases — the loss needs it.
    assert "dir_pred_enc" in o_tf and "dir_pred_enc" in o_pred
    assert o_tf["output"].shape == rgb.shape


def test_direction_head_autodisabled_under_categorical():
    m = _build(use_direction_head=True, direction_encoding="categorical")
    assert m.use_direction_head is False  # continuous-only; safe for the dirgen-categorical arm
    rgb = torch.rand(2, 3, 48, 48)
    with torch.no_grad():
        out = m(rgb, direction_id=torch.tensor([0, 1]))
    assert "dir_pred_enc" not in out


def test_direction_pred_loss_zero_on_perfect_and_masks_missing_meta():
    from losses.direction import direction_pred_loss
    from models.illum_embedding import IlluminationEmbedding

    phi = torch.tensor([0.3, 1.0])
    theta = torch.tensor([0.1, -0.5])
    b = torch.tensor([1.0, 2.0])
    gt = IlluminationEmbedding.encode_raw(phi, theta, b)
    # perfect prediction ⇒ ~0 loss
    assert direction_pred_loss(gt, phi, theta, b).item() < 1e-5
    # a corrupted sample with has_meta=False is ignored
    bad = gt.clone()
    bad[1] = 0.0
    has_meta = torch.tensor([True, False])
    assert direction_pred_loss(bad, phi, theta, b, has_meta).item() < 1e-5
