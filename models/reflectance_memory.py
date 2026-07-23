"""EMA reflectance memory bank (P2.2, optional).

A small `nn.Embedding(num_materials, dim)` that stores canonical
chromaticity-embedding vectors. At each training step:

- compute per-segment reflectance-embeddings from the network's R output;
- assign each segment to its nearest slot (cosine similarity);
- EMA-update the matched slot toward the segment embedding (running mean).

At inference, regions look up their nearest slot, providing a stable
material prior decoupled from the directly-predicted R map.

This is an *auxiliary* module: by default it does not feed back into the
decoder's R prediction during a vanilla forward pass. To inject it,
turn on `use_reflectance_memory` in the config and wire the slot embedding
through `DAINet`'s decoder — kept off by default in P2 scaffolding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReflectanceMemoryBank(nn.Module):
    def __init__(self, num_slots: int = 256, dim: int = 128, momentum: float = 0.99):
        super().__init__()
        self.num_slots = int(num_slots)
        self.dim = int(dim)
        self.momentum = float(momentum)
        # Embedding parameters are NOT trained by the optimizer; we update
        # them in-place via EMA. `register_buffer` keeps them in state_dict
        # and on the right device, but excludes them from grad.
        self.register_buffer("slots", torch.randn(num_slots, dim) * 0.01)
        self.register_buffer("counts", torch.zeros(num_slots))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        """EMA-update slots using a batch of [N, dim] embeddings.

        mask: optional [N] bool; only update from masked-in embeddings.
        """
        if mask is not None:
            embeddings = embeddings[mask]
        if embeddings.numel() == 0:
            return
        # Cosine-similarity assignment
        slots_n = nn.functional.normalize(self.slots, dim=-1)
        emb_n = nn.functional.normalize(embeddings, dim=-1)
        sims = emb_n @ slots_n.t()  # [N, num_slots]
        assign = sims.argmax(dim=-1)  # [N]
        # EMA per-slot
        for s in assign.unique().tolist():
            sel = assign == s
            new = embeddings[sel].mean(dim=0)
            self.slots[s] = self.momentum * self.slots[s] + (1.0 - self.momentum) * new
            self.counts[s] += int(sel.sum().item())

    @torch.no_grad()
    def lookup(self, embeddings: torch.Tensor) -> torch.Tensor:
        """For each query [N, dim], return the nearest slot embedding [N, dim]."""
        slots_n = nn.functional.normalize(self.slots, dim=-1)
        emb_n = nn.functional.normalize(embeddings, dim=-1)
        sims = emb_n @ slots_n.t()
        assign = sims.argmax(dim=-1)
        return self.slots[assign]
