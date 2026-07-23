"""Canonical paths + shared configuration for the dainet benchmark harness.

Single source of truth for dataset locations, repo locations, the python
interpreter, and the list of models / eval datasets. Importing this module
also injects the DAINet repo root onto ``sys.path`` so the benchmark
can reuse the project's own metric backbone (``evaluation.metrics``),
probe-mask builder (``data.probe_mask``) and LPIPS utility — guaranteeing
dainet's numbers in the benchmark match its paper exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
BEN_MODEL = Path(__file__).resolve().parents[2]          # .../DAINet/ben_model
BRACKETTOUCH = BEN_MODEL.parent                          # .../DAINet
REPOS = BEN_MODEL / "repos"
WEIGHTS = BEN_MODEL / "weights"
BACKBONES = WEIGHTS / "backbones"
CONFIGS = BEN_MODEL / "configs"
RESULTS = BEN_MODEL / "results"
DATA_PREP_OUT = BEN_MODEL / "data_prep" / "materialized"

# Make DAINet importable (evaluation.metrics, data.probe_mask, losses.*)
if str(BRACKETTOUCH) not in sys.path:
    sys.path.insert(0, str(BRACKETTOUCH))

# Python interpreter for subprocess training calls: reuse whatever interpreter is
# running this harness (i.e. the active env), so no hard-coded conda path is needed.
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Raw data
# ---------------------------------------------------------------------------
DATA_RAW = BRACKETTOUCH / "data" / "raw"

# Training source (MIT-MI): 985 scenes, 25 directions each, one shared GT.
MITMI_TRAIN_IN = DATA_RAW / "mit_mi" / "jpg" / "train"
MITMI_TRAIN_GT = DATA_RAW / "mit_mi" / "jpg_gt" / "train"

# Eval datasets (4).
MITMI_TEST_IN = DATA_RAW / "mit_mi" / "test" / "input" / "test"
MITMI_TEST_GT = DATA_RAW / "mit_mi" / "test" / "gt" / "test"
AMBIENT6K = DATA_RAW / "ben_data" / "ambient6k"
CL3AN = DATA_RAW / "ben_data" / "cl3an"
WSRD24 = DATA_RAW / "ben_data" / "wsrd24"

# ---------------------------------------------------------------------------
# Models + datasets registry
# ---------------------------------------------------------------------------
# Trained-from-scratch baselines + dainet (inference-only from its own ckpt).
# HVI-CIDNet was dropped from the benchmark: it collapsed to a mean-gray output
# on MIT-MI (~14.3 dB PSNR vs 23-25 dB for the other four) and never recovered
# even after the flagged grad-clip fix + retrain. Its weights/scripts/repo stay
# on disk but it is excluded from training, scoring and the comparison tables.
MODELS = ["restormer", "retinexformer", "rln2", "ifblend", "dainet"]
BASELINES = ["restormer", "retinexformer", "rln2", "ifblend"]

# Eval datasets. ``has_directions`` => 25 directions per scene.
DATASETS = {
    "mit_mi": {"has_directions": True, "has_probe_mask": True},
    "ambient6k": {"has_directions": False, "has_probe_mask": False},
    "cl3an": {"has_directions": False, "has_probe_mask": False},
    "wsrd24": {"has_directions": False, "has_probe_mask": False},
}

NET_HW = (512, 640)  # canonical working resolution (H, W) for mit_mi
N_DIRECTIONS = 25


def repo(name: str) -> Path:
    return REPOS / name


def full_weights_dir(model: str) -> Path:
    return WEIGHTS / f"{model}_mitmi"
