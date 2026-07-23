"""Encode the per-direction illumination conditioning.

Two interchangeable encodings (``encoding=``):

- ``"continuous"`` (default, the contribution-1 claim) — the per-direction
  ``(phi, theta, brightness_norm)`` triple is featurised as
  ``[sin φ, cos φ, sin θ, cos θ, log(bnorm)]`` and mapped through a 3-layer
  MLP. Generalises to unseen directions; inference falls back on a learned
  null token.

- ``"categorical"`` — a plain ``nn.Embedding(num_directions, embed_dim)``
  indexed by ``direction_id`` (0..24). This is the categorical baseline the
  direction-generalisation ablation compares against: it cannot interpolate to
  held-out directions. The embedding is **zero-initialised** so identity-at-init
  is preserved (the downstream FiLM / fusion consumers are zero-init too, but a
  zero illum embedding keeps the categorical and continuous paths comparable at
  step 0).

Both paths expose the same ``forward(phi, theta, bnorm, direction_id)`` so the
network does not branch on the encoding.
"""

from __future__ import annotations

import torch
import torch.nn as nn


_VALID_ENCODINGS = ("continuous", "categorical")


class IlluminationEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        encoding: str = "continuous",
        num_directions: int = 25,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding = str(encoding).lower()
        if self.encoding not in _VALID_ENCODINGS:
            raise ValueError(
                f"encoding must be one of {_VALID_ENCODINGS}, got {encoding!r}"
            )
        if self.encoding == "continuous":
            self.mlp = nn.Sequential(
                nn.Linear(5, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, embed_dim),
            )
        else:  # categorical
            self.embedding = nn.Embedding(int(num_directions), embed_dim)
            nn.init.zeros_(self.embedding.weight)

    @staticmethod
    def encode_raw(phi: torch.Tensor, theta: torch.Tensor, bnorm: torch.Tensor) -> torch.Tensor:
        bnorm_clamped = torch.clamp(bnorm, min=1e-6)
        return torch.stack(
            [
                torch.sin(phi),
                torch.cos(phi),
                torch.sin(theta),
                torch.cos(theta),
                torch.log(bnorm_clamped),
            ],
            dim=-1,
        )

    def forward_encoded(self, enc: torch.Tensor) -> torch.Tensor:
        """Map a pre-built ``[B, 5]`` encoding straight through the MLP.

        Used by the light-direction head (``models/direction_head.py``), whose
        prediction already has the ``encode_raw`` layout
        ``[sin φ, cos φ, sin θ, cos θ, log b]``. Continuous encoding only.
        """
        if self.encoding != "continuous":
            raise ValueError("forward_encoded is only valid for continuous encoding")
        return self.mlp(enc)  # [B, embed_dim]

    def forward(
        self,
        phi: torch.Tensor,
        theta: torch.Tensor,
        bnorm: torch.Tensor,
        direction_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoding == "categorical":
            if direction_id is None:
                raise ValueError(
                    "categorical illumination encoding requires direction_id"
                )
            return self.embedding(direction_id.long())
        x = self.encode_raw(phi, theta, bnorm)  # [B, 5]
        return self.mlp(x)  # [B, embed_dim]
