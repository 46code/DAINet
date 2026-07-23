"""Dedicated surface-normals encoder with multi-scale feature fusion.

This is the stronger alternative to the pre-fuse 1×1 conv (see
``models/encoder.py``): instead of collapsing RGB ⊕ normals to 3 channels
*before* the ConvNeXt backbone (which throws away the timm pretrained stem's
RGB specialisation), the normals get their **own** small CNN that produces a
feature map at each ConvNeXt stride. Those features are added — through a
**zero-initialised** 1×1 projection — onto the corresponding backbone stage
output. So:

- The ConvNeXt backbone stays pure-RGB (pretrained weights untouched).
- Normals get real spatial capacity at every scale (shading is ``n · ω`` and
  is inherently multi-scale / spatial), not a single 3-channel bottleneck.
- The zero-init projections make every fused residual **exactly zero at step
  0**, so the whole network is bit-exactly RGB-only at init and learns to mix
  in geometry over training (identity-at-init contract preserved).

When normals are absent at forward time (``normals=None`` /
``has_normals=False``), the caller passes zeros; the encoder still runs and
applies its trained "no-normals" response (consistent with the pre-fuse path).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _down(in_ch: int, out_ch: int) -> nn.Sequential:
    """A stride-2 conv block: halves H, W."""
    groups = min(8, out_ch)
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1),
        nn.GroupNorm(groups, out_ch),
        nn.GELU(),
    )


class NormalsEncoder(nn.Module):
    """[B, 3, H, W] surface normals → per-stage additive feature residuals.

    Produces one residual per backbone stage (strides 4, 8, 16, 32), each
    matched to that stage's channel count via a zero-init 1×1 projection so
    the residual is the identity (zero) at step 0.
    """

    def __init__(
        self,
        feature_channels: list[int],
        feature_strides: list[int] | None = None,
        in_channels: int = 3,
        hidden: int = 32,
    ):
        super().__init__()
        self.feature_channels = list(feature_channels)
        # Trunk: progressive stride-2 downsampling. tap[i] sits at stride
        # 4·2^i, i.e. the ConvNeXt stage strides [4, 8, 16, 32].
        self.stem = _down(in_channels, hidden)          # stride 2
        self.down_s4 = _down(hidden, hidden)            # stride 4   -> tap 0
        self.down_s8 = _down(hidden, hidden * 2)        # stride 8   -> tap 1
        self.down_s16 = _down(hidden * 2, hidden * 2)   # stride 16  -> tap 2
        self.down_s32 = _down(hidden * 2, hidden * 4)   # stride 32  -> tap 3
        tap_in = [hidden, hidden * 2, hidden * 2, hidden * 4]

        # Zero-init 1×1 projections to each backbone stage's channel count.
        # Zero weight + bias => the fused residual is exactly 0 at step 0.
        self.projs = nn.ModuleList(
            [nn.Conv2d(tap_in[i], self.feature_channels[i], kernel_size=1) for i in range(4)]
        )
        for proj in self.projs:
            nn.init.zeros_(proj.weight)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)

    def forward(self, normals: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(normals)
        f4 = self.down_s4(x)
        f8 = self.down_s8(f4)
        f16 = self.down_s16(f8)
        f32 = self.down_s32(f16)
        taps = [f4, f8, f16, f32]
        return [proj(t) for proj, t in zip(self.projs, taps)]
