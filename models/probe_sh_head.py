"""Map the illumination embedding to per-channel real-SH coefficients.

Supervised by chrome-probe SH targets (`data/probe_sh.chrome_to_sh`). Small
init keeps the loss bounded at step 0.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ProbeSHHead(nn.Module):
    def __init__(self, embed_dim: int = 128, l_max: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.l_max = l_max
        self.n_coeff = (l_max + 1) ** 2
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3 * self.n_coeff),
        )
        # Small init on the final layer so step-0 sh_pred is near zero.
        nn.init.normal_(self.mlp[-1].weight, std=1e-3)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, illum_emb: torch.Tensor) -> torch.Tensor:
        flat = self.mlp(illum_emb)
        return flat.view(-1, 3, self.n_coeff)
