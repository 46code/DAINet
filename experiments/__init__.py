"""dainet experiment registry — named ablations as override patches.

Each experiment is a sparse override dict deep-merged onto the base config
(``configs/dainet.yaml``) by ``scripts/run_experiment.py``, which routes all
outputs to ``runs/<name>/`` and writes a fully-resolved ``config.yaml``
snapshot there so every run is reproducible from its own committed file.

See ``docs/dainet_ablation_plan.md`` for the experiment grid and
``docs/dainet_reporting.md`` for the run → report → compare workflow.
"""

from .registry import (
    PRIORITY_ORDER,
    REGISTRY,
    experiments_by_priority,
    get_experiment,
    list_experiments,
)

__all__ = [
    "PRIORITY_ORDER",
    "REGISTRY",
    "experiments_by_priority",
    "get_experiment",
    "list_experiments",
]
