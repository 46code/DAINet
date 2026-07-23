"""Identity-at-init invariants for the 2026-05-29 architecture modules.

The illumination chroma-field and the global illuminant head must NOT break
the identity-at-init contract: with both on, ``I_out == input_rgb`` at step 0,
``R ∈ [0,1]``, ``L`` within the tanh-exp bounds, the chroma field ≡ 1, and the
illuminant prediction is achromatic (equal channels).
"""

from __future__ import annotations

import math

import torch

from models.network import DAINet


def _build(**kw):
    return DAINet(
        embed_dim=64, seg_embed_dim=64, illum_hidden=64, attn_heads=4,
        pretrained_encoder=False, backbone="convnext_tiny",
        use_swin_bottleneck=True, use_illum_token=True, use_normals=True,
        **kw,
    )


def test_identity_at_init_with_new_modules():
    torch.manual_seed(0)
    m = _build(use_illum_chroma_field=True, use_illuminant_head=True).eval()
    rgb = torch.rand(2, 3, 64, 80)
    with torch.no_grad():
        out = m(rgb)
    assert (out["output"] - rgb).abs().max().item() < 1e-3, "identity-at-init broken"
    assert out["reflectance"].min() >= 0.0 and out["reflectance"].max() <= 1.0
    lo, hi = math.exp(-2.5), math.exp(2.5)
    assert out["illumination"].min() >= lo - 1e-3
    assert out["illumination"].max() <= hi + 1e-3


def test_chroma_field_is_unity_at_init():
    # With the field ≡ 1 at init, turning it on must not change L vs off.
    torch.manual_seed(0)
    rgb = torch.rand(2, 3, 64, 80)
    m_off = _build(use_illum_chroma_field=False).eval()
    torch.manual_seed(0)
    m_on = _build(use_illum_chroma_field=True).eval()
    # Copy shared weights so only the (zero-init) field differs.
    m_on.load_state_dict(m_off.state_dict(), strict=False)
    with torch.no_grad():
        L_off = m_off(rgb)["illumination"]
        L_on = m_on(rgb)["illumination"]
    assert (L_on - L_off).abs().max().item() < 1e-4


def test_illuminant_head_is_neutral_at_init():
    torch.manual_seed(0)
    m = _build(use_illuminant_head=True).eval()
    with torch.no_grad():
        out = m(torch.rand(2, 3, 48, 48))
    illum = out["illuminant"]
    assert illum.shape == (2, 3)
    # equal channels ⇒ achromatic direction
    assert (illum - illum.mean(dim=1, keepdim=True)).abs().max().item() < 1e-5
