"""Named experiment registry for dainet: the full baseline + the ablation suite.

Every entry is ``{"description": str, "priority": str, "group": str,
"overrides": <nested dict>}``. The ``overrides`` are deep-merged onto
``configs/dainet.yaml``; only the keys that differ are listed, so each ablation's
intent is obvious at a glance and the grid stays in sync with the base config
automatically.

The base config (`configs/dainet.yaml`) is the **maximal-capacity full model**
(`dainet_full`): every architectural component on, every optional head supervised
by its loss. Each ablation removes EXACTLY ONE component — the literature-standard
drop-one-from-full-model design — except two deliberately-multi-knob entries
(see ``_MULTI_KNOB_WHITELIST``): the bundled material pathway and the compound
spatial-prior baseline. ``validate_registry()`` enforces this.

The thesis asks how **directional-lighting cues** and **spatial priors** help
produce a flat-lit, illumination-normalized image free of lighting gradients in
the final RGB. The suite is exactly these 7 ablations, grouped along those two
axes (the reporting groups used by ``scripts/compare_ablations.py``):

  group                run                    question / mechanism dropped
  directional lighting abl_no_xdir            cross-direction relighting consistency
  directional lighting abl_no_dirhead         learned (φ,θ,b) direction head (inference)
  directional lighting abl_single_direction   value of 25-direction supervision
  priors               abl_no_material        material aux supervision (novelty C)
  priors               abl_singleview_sam     multi-view SAM2 fusion (novelty A)
  priors               abl_no_normals         geometry / normals encoder (novelty D)
  priors               abl_no_spatial_priors  compound prior removal (normals+SAM+material)

``priority`` ∈ {high, moderate} drives run order. Convention: an experiment
``<name>`` writes everything under ``runs/<name>/`` and sets
``wandb.run_name = <name>``. Add an ablation by adding one entry here (and a
group); the one-knob guard keeps single-component runs honest.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


# Priority tiers, in run order.
PRIORITY_ORDER = ("high", "moderate", "low")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BASE_CONFIG = _REPO_ROOT / "configs" / "dainet.yaml"


REGISTRY: dict[str, dict[str, Any]] = {
    # ===================== run of record =====================
    "dainet_full": {
        "priority": "high",
        "group": "baseline",
        "description": (
            "Maximal-capacity full model: ConvNeXt-Base + SwinV2 + normals "
            "encoder + all heads (material/SH/specular/illuminant/chroma-field) "
            "+ curated role-balanced loss stack. The baseline every ablation "
            "drops one component from."
        ),
        "overrides": {},
    },

    # ============ AXIS 1: DIRECTIONAL LIGHTING (drop a direction cue) ========
    "abl_no_xdir": {
        "priority": "high",
        "group": "directional lighting",
        "description": "Drop xdir_relight — cross-direction relighting consistency (R·L identifiability across directions). HEADLINE novelty B.",
        "overrides": {"loss": {"xdir_relight": 0.0}},
    },
    "abl_no_dirhead": {
        "priority": "high",
        "group": "directional lighting",
        "description": (
            "Remove the learned (φ,θ,b) direction head (use_direction_head=false) — inference "
            "falls back to null_illum_emb + illum_token. Measures the head's contribution."
        ),
        "overrides": {"model": {"use_direction_head": False}},
    },
    "abl_single_direction": {
        "priority": "high",
        "group": "directional lighting",
        "description": (
            "single direction/scene — tests value of 25-direction supervision; "
            "cross-direction losses naturally inactive"
        ),
        # One knob: directions_per_scene=1 switches the train loader to the
        # SingleDirectionPerSceneSampler. With no same-scene pairs in a batch,
        # xdir_relight and dir_consistency_R go to zero by construction — we do
        # NOT zero them explicitly (that's the whole point of the single knob).
        "overrides": {"dataset": {"directions_per_scene": 1}},
    },

    # ============ AXIS 2: SPATIAL PRIORS (drop a prior pathway) ==============
    "abl_no_material": {
        "priority": "high",
        "group": "priors",
        "description": "Remove the material aux pathway (head + material_ce + material_R_var) — novelty C.",
        "overrides": {
            "model": {"use_material_head": False},
            "loss": {"material_ce": 0.0, "material_R_var": 0.0},
        },
    },
    "abl_singleview_sam": {
        "priority": "high",
        "group": "priors",
        "description": (
            "Single-view SAM ids (no 25-view fusion) — novelty A. Point paths.sam_root at "
            "the single-view cache built by precompute_sam.py --n_views 1."
        ),
        "overrides": {"paths": {"sam_root": "data/raw/mit_mi/sam_masks_sv"}},
    },
    "abl_no_normals": {
        "priority": "high",
        "group": "priors",
        "description": "Remove the dedicated normals encoder (normals_fusion=none) — novelty D.",
        # One knob: with normals_fusion set explicitly the encoder ignores
        # use_normals entirely (see models/encoder._resolve_normals_fusion).
        "overrides": {"model": {"normals_fusion": "none"}},
    },
    "abl_no_spatial_priors": {
        "priority": "high",
        "group": "priors",
        "description": (
            "no normals/SAM/material — tests compound spatial-prior contribution beyond "
            "direction-aware decomposition"
        ),
        # Multi-knob by design (whitelisted): drops the geometry, SAM-FiLM and
        # material-aux pathways together, leaving the RGB encoder + direction
        # conditioning (MLP + head) + the xdir/dir_consistency/probe_sh/retinex
        # R·L decomposition intact.
        "overrides": {
            "model": {
                "normals_fusion": "none",
                "use_sam_conditioning": False,
                "use_material_head": False,
            },
            "loss": {"material_ce": 0.0, "material_R_var": 0.0},
        },
    },
}


# The default suite = every run in the registry (dainet_full is the baseline).
DEFAULT_SUITE: tuple[str, ...] = tuple(
    n for n in REGISTRY if n != "dainet_full"
)

# Runs allowed to change more than one leaf key vs the base config: the bundled
# material pathway and the compound spatial-prior baseline. Everything else must
# be a single-knob drop-one ablation.
_MULTI_KNOB_WHITELIST = frozenset(
    {
        "dainet_full",
        # bundled material pathway (head + material_ce + material_R_var)
        "abl_no_material",
        # compound prior removal, reported as the RQ2 prior-contribution baseline
        "abl_no_spatial_priors",
    }
)


def list_experiments() -> list[tuple[str, str]]:
    """(name, description) pairs in registry order."""
    return [(name, spec["description"]) for name, spec in REGISTRY.items()]


def default_runs() -> list[str]:
    """The ablation names that make up the default suite (no baseline)."""
    return list(DEFAULT_SUITE)


def run_group(name: str) -> str:
    """Reporting group for a run, along the two thesis axes
    (baseline / directional lighting / priors)."""
    return REGISTRY.get(name, {}).get("group", "other")


def experiments_by_priority(
    priority: str | None = None,
) -> list[tuple[str, str, str]]:
    """(name, priority, description) triples, grouped in PRIORITY_ORDER.

    Pass ``priority`` to return only one tier (high|moderate|low).
    """
    if priority is not None and priority not in PRIORITY_ORDER:
        raise ValueError(
            f"Unknown priority {priority!r}. Choose from {PRIORITY_ORDER}."
        )
    out: list[tuple[str, str, str]] = []
    for tier in PRIORITY_ORDER:
        if priority is not None and tier != priority:
            continue
        for name, spec in REGISTRY.items():
            if spec.get("priority", "low") == tier:
                out.append((name, tier, spec["description"]))
    return out


def get_experiment(name: str) -> dict[str, Any]:
    """Return a deep copy of the experiment spec, or raise with suggestions."""
    if name not in REGISTRY:
        avail = ", ".join(REGISTRY.keys())
        raise KeyError(f"Unknown experiment {name!r}. Available: {avail}")
    return deepcopy(REGISTRY[name])


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested dict to dotted leaf keys."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def validate_registry(base_config_path: str | Path | None = None) -> dict[str, int]:
    """One-knob guard: assert every single-component ablation differs from the
    base config by EXACTLY ONE leaf key.

    Bundled / protocol runs in ``_MULTI_KNOB_WHITELIST`` are exempt. Raises
    ``AssertionError`` on a violation; returns ``{run: n_changed_keys}`` on
    success. Called at runner start and covered by tests/test_registry_one_knob.py.
    """
    base_config_path = Path(base_config_path) if base_config_path else _BASE_CONFIG
    base_flat = _flatten(yaml.safe_load(Path(base_config_path).read_text()))
    counts: dict[str, int] = {}
    for name, spec in REGISTRY.items():
        overrides_flat = _flatten(spec.get("overrides", {}))
        changed = [k for k, v in overrides_flat.items() if base_flat.get(k) != v]
        counts[name] = len(changed)
        if name in _MULTI_KNOB_WHITELIST:
            continue
        assert len(changed) == 1, (
            f"Single-component run {name!r} must change exactly one config knob "
            f"vs {base_config_path}; it changes {len(changed)}: {changed}. "
            f"Reduce it to one knob or add it to _MULTI_KNOB_WHITELIST."
        )
    return counts
