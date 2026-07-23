"""ConvNeXt feature extractor, RGB-only backbone, with per-stage FiLM.

The encoder runs a pure sRGB image through the ConvNeXt backbone. Segmentation
(SAM2), when available, conditions the encoder via FiLM applied AFTER each
stage — modulating how features behave rather than mixing into the pixel
stream. The FiLM γ, β projection is zero-initialized so segmentation is the
identity at step 0.

Backbone is timm-keyed via ``backbone=``:
- ``convnext_tiny``  (~28M, ablation)
- ``convnext_base``  (~88M, full-capacity default)
The downstream code consumes ``feature_channels`` directly from
``backbone.feature_info.channels()`` so swapping the backbone does not
require manual dim plumbing in the decoder / fusion.

Surface-normals fusion (``normals_fusion``):
- ``"none"``    — pure RGB; no normals path.
- ``"prefuse"`` — a tiny 1×1 conv mixes RGB ⊕ normals (6→3) *before* the
  backbone. Identity-on-RGB / zero-on-normals init keeps the model RGB-only
  at step 0. Cheap but collapses normals to 3 channels up front.
- ``"encoder"`` — a dedicated multi-scale ``NormalsEncoder`` (see
  ``models/normals_encoder.py``) adds a zero-init residual onto each backbone
  stage. Backbone stays pure-RGB (pretrained weights untouched); normals get
  real spatial capacity at every scale. This is the full-capacity default.

In every mode, when ``normals=None`` is passed at forward time the encoder
substitutes zeros so the trained weights still apply (bypassing would silently
change inference behaviour).
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .decoder import FiLM
from .normals_encoder import NormalsEncoder


_VALID_NORMALS_FUSION = ("none", "prefuse", "encoder")


def _resolve_normals_fusion(normals_fusion: str | None, use_normals: bool) -> str:
    """Back-compat: ``use_normals`` (legacy bool) maps to a fusion mode."""
    if normals_fusion is None:
        return "prefuse" if use_normals else "none"
    mode = str(normals_fusion).lower()
    if mode not in _VALID_NORMALS_FUSION:
        raise ValueError(
            f"normals_fusion must be one of {_VALID_NORMALS_FUSION}, got {mode!r}"
        )
    return mode


class ConvNextEncoder(nn.Module):
    """Returns four FiLM-modulated feature maps at strides 4, 8, 16, 32."""

    def __init__(
        self,
        seg_embed_dim: int = 128,
        pretrained: bool = True,
        backbone: str = "convnext_tiny",
        use_normals: bool = False,
        normals_fusion: str | None = None,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.normals_fusion = _resolve_normals_fusion(normals_fusion, use_normals)
        # Legacy attribute kept truthy whenever any normals path is active.
        self.use_normals = self.normals_fusion != "none"

        # Load model, falling back to non-pretrained if cache is unavailable.
        # Network access is disabled via HF_HUB_OFFLINE env var to prevent hangs.
        try:
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                features_only=True,
                in_chans=3,
            )
        except (FileNotFoundError, RuntimeError) as e:
            if pretrained:
                import warnings
                warnings.warn(
                    f"Failed to load pretrained {backbone} (cache not found or offline: {e}). "
                    f"Using non-pretrained initialization.",
                    RuntimeWarning
                )
                self.backbone = timm.create_model(
                    backbone,
                    pretrained=False,
                    features_only=True,
                    in_chans=3,
                )
            else:
                raise
        # ConvNeXt-Tiny: [96, 192, 384, 768]  at strides [4, 8, 16, 32]
        # ConvNeXt-Base: [128, 256, 512, 1024] at strides [4, 8, 16, 32]
        self.feature_channels: list[int] = list(self.backbone.feature_info.channels())
        self.feature_strides: list[int] = list(self.backbone.feature_info.reduction())
        self.films = nn.ModuleList(
            [FiLM(embed_dim=seg_embed_dim, feat_dim=c) for c in self.feature_channels]
        )

        if self.normals_fusion == "prefuse":
            # 6-channel (rgb ⊕ normals) → 3-channel for the ConvNeXt stem.
            # Identity-on-RGB / zero-on-normals init so the network is
            # bit-exactly equivalent to RGB-only at step 0.
            self.pre_fuse = nn.Conv2d(6, 3, kernel_size=1, bias=False)
            with torch.no_grad():
                w = torch.zeros(3, 6, 1, 1)
                w[0, 0, 0, 0] = 1.0
                w[1, 1, 0, 0] = 1.0
                w[2, 2, 0, 0] = 1.0
                self.pre_fuse.weight.copy_(w)
        elif self.normals_fusion == "encoder":
            # Dedicated multi-scale normals encoder; zero-init projections make
            # every fused residual exactly 0 at step 0 (RGB-only at init).
            self.normals_encoder = NormalsEncoder(
                feature_channels=self.feature_channels,
                feature_strides=self.feature_strides,
                in_channels=3,
            )

    def forward(
        self,
        rgb: torch.Tensor,
        seg_emb: torch.Tensor,
        normals: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        if self.normals_fusion == "prefuse":
            # Zeros when normals are absent so the trained RGB-channel weights
            # still apply and the conv runs in the same shape.
            n = normals if normals is not None else torch.zeros_like(rgb)
            x = self.pre_fuse(torch.cat([rgb, n.to(dtype=rgb.dtype)], dim=1))
            feats = self.backbone(x)
        elif self.normals_fusion == "encoder":
            feats = self.backbone(rgb)
            n = normals if normals is not None else torch.zeros_like(rgb)
            nfeats = self.normals_encoder(n.to(dtype=rgb.dtype))
            # Resize each normals residual to the exact backbone stage size —
            # the two downsampling paths can differ by ±1 px on odd dims.
            # (projections are zero-init, so this is the identity at step 0.)
            fused = []
            for f, nf in zip(feats, nfeats):
                if nf.shape[-2:] != f.shape[-2:]:
                    nf = F.interpolate(nf, size=f.shape[-2:], mode="bilinear", align_corners=False)
                fused.append(f + nf)
            feats = fused
        else:  # "none"
            feats = self.backbone(rgb)
        return [film(f, seg_emb) for f, film in zip(feats, self.films)]


# Back-compat alias so existing imports continue to resolve.
ConvNextTinyEncoder = ConvNextEncoder
