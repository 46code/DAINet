"""Global illuminant estimation head (training-only auxiliary).

Reads the illumination embedding and predicts a single 3-vector illuminant
chromaticity — an estimate of the colour of the light the model must remove.
Supervised by ``losses.illuminant.illuminant_angular_loss`` against the
per-image input/target mean-RGB ratio (the gray-world lighting-colour
estimate), in angular space.

This is a *direct* lever for color constancy: rather than hoping the
reconstruction loss implicitly neutralises the cast, the head is optimised
straight against illuminant direction. It is an auxiliary head only —
nothing downstream consumes ``out["illuminant"]`` — so it never
changes ``I_out`` and is safe to leave on or off.

Identity-friendly init: the final linear is zero-init, so every channel
emits ``softplus(0) = ln 2`` at step 0 → a perfectly neutral (achromatic)
illuminant direction, contributing no gradient bias before training.

On classifier-free-dropout batches the embedding is the learned null token,
so the head then learns the dataset-mean illuminant prior — a benign
regulariser, never harmful (the head output is unused at inference).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminantHead(nn.Module):
    def __init__(self, embed_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or embed_dim
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 3)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, illum_emb: torch.Tensor) -> torch.Tensor:
        raw = self.fc2(self.act(self.fc1(illum_emb)))  # [B, 3], zero at init
        # softplus → strictly positive chromaticity; equal channels at init.
        return F.softplus(raw) + 1e-3
