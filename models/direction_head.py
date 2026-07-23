"""Light-direction prediction head (contribution C, 2026-06-08).

Predicts the per-image illumination direction ``(φ, θ, b)`` from the encoder
bottleneck so the model can condition on lighting **without** the capture-rig
metadata that only exists in the calibrated dataset. At inference this replaces
the learned ``null_illum_emb``: the predicted direction is fed through the SAME
``IlluminationEmbedding`` MLP the GT ``(φ, θ, b)`` uses during training, so the
deploy path matches the trained conditioning distribution.

Output layout matches ``IlluminationEmbedding.encode_raw`` exactly —
``[sin φ, cos φ, sin θ, cos θ, log b]`` — so the regression loss can build its
target with that same helper, and ``IlluminationEmbedding.forward_encoded`` can
consume the prediction directly.

Each ``(sin, cos)`` pair is L2-normalised to lie on the unit circle (a valid
angle); ``log b`` is unconstrained. The final linear is initialised to the
**neutral** direction ``(φ=0, θ=0, b=1) → [0, 1, 0, 1, 0]`` (zero weight, neutral
bias) rather than degenerate zeros, so the pair-normalisation is well-defined
from step 0. Identity-at-init of the whole network is preserved by the trainer
teacher-forcing GT / null conditioning on the main path at step 0 — this head
only feeds the regression loss and the inference / mixed-forcing path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DirectionHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or feat_dim
        self.norm = nn.GroupNorm(1, feat_dim)
        self.fc1 = nn.Linear(feat_dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, 5)
        # Neutral init: predict (φ=0, θ=0, b=1) ⇒ encoding [0, 1, 0, 1, 0].
        nn.init.zeros_(self.fc2.weight)
        with torch.no_grad():
            self.fc2.bias.copy_(torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0]))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B, C, H, W] → predicted encoding [B, 5] = [sinφ,cosφ,sinθ,cosθ,logb]."""
        x = self.norm(feats)
        x = x.mean(dim=(-2, -1))  # global average pool → [B, C]
        raw = self.fc2(self.act(self.fc1(x)))  # [B, 5]
        phi_pair = F.normalize(raw[:, 0:2], dim=-1, eps=1e-6)
        theta_pair = F.normalize(raw[:, 2:4], dim=-1, eps=1e-6)
        log_b = raw[:, 4:5]
        return torch.cat([phi_pair, theta_pair, log_b], dim=-1)  # [B, 5]
