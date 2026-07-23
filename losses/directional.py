"""Cross-direction reflectance invariance.

For every same-scene pair (i, j) in the batch (matched by string equality of
the `scene` field), penalize the L1 distance between the predicted
reflectance maps R[i] and R[j]. The ScenePairBatchSampler guarantees at
least one such pair per batch.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def directional_R_consistency(
    R: torch.Tensor, scenes: list[str]
) -> torch.Tensor:
    if len(scenes) < 2:
        return torch.zeros((), device=R.device, dtype=R.dtype)
    losses: list[torch.Tensor] = []
    for i in range(len(scenes)):
        for j in range(i + 1, len(scenes)):
            if scenes[i] == scenes[j]:
                losses.append(F.l1_loss(R[i], R[j]))
    if not losses:
        return torch.zeros((), device=R.device, dtype=R.dtype)
    return torch.stack(losses).mean()
