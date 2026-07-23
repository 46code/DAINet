"""The ablation registry is exactly dainet_full + 7 runs, and every
single-component ablation differs from the base config by exactly one knob.

This locks the drop-one-from-full-model design: a reviewer can trust that
`abl_no_xdir` vs `dainet_full` isolates one variable. The runs sit on two thesis
axes (directional lighting / priors). Bundled / compound runs (the material
pathway, the compound spatial-prior baseline) are whitelisted.
"""

from __future__ import annotations

import pytest

from experiments import registry as R


EXPECTED_SUITE = {
    # directional lighting
    "abl_no_xdir",
    "abl_no_dirhead",
    "abl_single_direction",
    # priors
    "abl_no_material",
    "abl_singleview_sam",
    "abl_no_normals",
    "abl_no_spatial_priors",
}


def test_registry_is_baseline_plus_seven():
    assert set(R.REGISTRY) == EXPECTED_SUITE | {"dainet_full"}
    assert set(R.default_runs()) == EXPECTED_SUITE
    # The deleted runs must be gone.
    for gone in ("abl_no_swin", "abl_no_illum_token", "abl_nullcond0",
                 "abl_no_specular", "abl_dists_add", "abl_subset30",
                 "abl_no_probe_sh", "abl_no_retinex",
                 "abl_dirgen_continuous", "abl_dirgen_categorical"):
        assert gone not in R.REGISTRY


def test_one_knob_guard_passes():
    counts = R.validate_registry()
    assert counts["dainet_full"] == 0
    # Single-component runs change exactly one leaf key.
    for name in ("abl_no_xdir", "abl_no_normals", "abl_singleview_sam",
                 "abl_no_dirhead", "abl_single_direction"):
        assert counts[name] == 1, (name, counts[name])


def test_every_run_has_a_group():
    for name in R.REGISTRY:
        assert R.run_group(name) in {
            "baseline", "directional lighting", "priors",
        }


def test_guard_rejects_a_second_knob(monkeypatch):
    """Injecting a second knob into a single-component run must raise."""
    spec = dict(R.REGISTRY["abl_no_xdir"])
    spec["overrides"] = {"loss": {"xdir_relight": 0.0}, "model": {"use_swin_bottleneck": False}}
    patched = dict(R.REGISTRY, abl_no_xdir=spec)
    monkeypatch.setattr(R, "REGISTRY", patched)
    with pytest.raises(AssertionError):
        R.validate_registry()
