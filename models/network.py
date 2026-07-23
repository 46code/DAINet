"""DAINet — full model composition.

Public API is intentionally minimal: ``model(rgb)`` is the only call required
at inference / test time. Training supplies optional conditioning via the
kwargs ``seg``, ``phi``, ``theta``, ``bnorm``; when any group is omitted, the
model substitutes a learned null embedding (no zero-filled dummy tensors).

Inputs (training):
    rgb:   [B, 3, H, W] in [0, 1]
    seg:   [B, 1, H, W] int32 segment ids   (optional)
    phi, theta, bnorm: [B] floats           (optional, all-or-nothing)

Inputs (test / inference):
    rgb only. SAM seg + DSINE normals are computed from the RGB (cache/inline);
    illumination is supplied by the learned ``illum_token`` and, when
    ``use_direction_head`` is on, a predicted ``(φ, θ, b)`` routed through
    ``illum_embed`` (replacing ``null_illum_emb``). ``seg``/``normals`` are still
    optional kwargs — when omitted the learned null seg token / zero-normals apply.

Outputs (dict):
    output:        [B, 3, H, W] in [0, 1] — final corrected image
    reflectance:   [B, 3, H, W] — direction-invariant reflectance R
    illumination:  [B, 3, H, W] — directional illumination L  (R * L = I_out)
    sh_pred:       [B, 3, n_sh] — predicted chrome-SH coefficients
    illum_emb:     [B, embed_dim] — illumination conditioning vector used
    seg_emb:       [B, seg_embed_dim] — segmentation conditioning vector used
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import DualHeadDecoder
from .direction_head import DirectionHead
from .encoder import ConvNextEncoder
from .fusion import CrossAttentionFusion
from .illum_embedding import IlluminationEmbedding
from .illum_token import LatentIlluminationToken
from .material_head import MaterialHead
from .probe_sh_head import ProbeSHHead
from .reflectance_memory import ReflectanceMemoryBank
from .seg_encoder import SegmentationEncoder
from .specular_head import SpecularHead
from .swin_bottleneck import SwinBottleneck


class DAINet(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        seg_embed_dim: int = 128,
        illum_hidden: int = 128,
        attn_heads: int = 8,
        sh_l_max: int = 4,
        pretrained_encoder: bool = True,
        backbone: str = "convnext_tiny",
        use_swin_bottleneck: bool = False,
        use_specular_head: bool = False,
        use_illum_token: bool = False,
        use_reflectance_memory: bool = False,
        reflectance_memory_size: int = 256,
        use_material_head: bool = False,
        material_num_classes: int = 0,
        material_head_hidden: int = 256,
        use_normals: bool = False,
        normals_fusion: str | None = None,
        use_sam_conditioning: bool = True,
        direction_encoding: str = "continuous",
        num_directions: int = 25,
        use_illum_chroma_field: bool = False,
        use_illuminant_head: bool = False,
        use_direction_head: bool = False,
        activation_checkpoint: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.seg_embed_dim = seg_embed_dim
        self.sh_l_max = sh_l_max
        self.use_swin_bottleneck = bool(use_swin_bottleneck)
        self.use_specular_head = bool(use_specular_head)
        self.use_illum_token = bool(use_illum_token)
        self.use_reflectance_memory = bool(use_reflectance_memory)
        self.use_material_head = bool(use_material_head) and int(material_num_classes) > 0
        self.material_num_classes = int(material_num_classes)
        self.direction_encoding = str(direction_encoding).lower()
        self.activation_checkpoint = bool(activation_checkpoint)

        self.encoder = ConvNextEncoder(
            seg_embed_dim=seg_embed_dim,
            pretrained=pretrained_encoder,
            backbone=backbone,
            use_normals=bool(use_normals),
            normals_fusion=normals_fusion,
        )
        # Truth lives on the encoder now (handles the use_normals→mode mapping).
        self.normals_fusion = self.encoder.normals_fusion
        self.use_normals = self.encoder.use_normals
        # SAM/seg FiLM conditioning. When off, the encoder always sees the
        # learned null_seg_emb even if a seg map is provided — the spatial-prior
        # ablation (`abl_no_spatial_priors`) uses this to drop SAM conditioning.
        self.use_sam_conditioning = bool(use_sam_conditioning)
        self.use_illum_chroma_field = bool(use_illum_chroma_field)
        self.use_illuminant_head = bool(use_illuminant_head)
        # Light-direction head is continuous-encoding only (it feeds the
        # IlluminationEmbedding MLP). Auto-disable under categorical encoding so
        # the dirgen-categorical ablation never hits an invalid combination.
        self.use_direction_head = bool(use_direction_head) and self.direction_encoding == "continuous"
        self.seg_encoder = SegmentationEncoder(embed_dim=seg_embed_dim)
        self.illum_embed = IlluminationEmbedding(
            embed_dim=embed_dim,
            hidden_dim=illum_hidden,
            encoding=self.direction_encoding,
            num_directions=int(num_directions),
        )

        # Learned null conditioning tokens. Zero-init keeps the identity-at-init
        # contract: a zero seg_emb produces zero (γ, β) in the encoder FiLM,
        # a zero illum_emb produces zero (γ, β) in the decoder FiLM and zero
        # attention residual in the cross-attention fusion.
        self.null_seg_emb = nn.Parameter(torch.zeros(1, seg_embed_dim))
        self.null_illum_emb = nn.Parameter(torch.zeros(1, embed_dim))

        feat_channels = self.encoder.feature_channels
        # P6.2 — SwinV2 self-attention bottleneck on s4 (residual). Off by
        # default; turn on via model.use_swin_bottleneck for the full cfg.
        if self.use_swin_bottleneck:
            self.swin_bottleneck = SwinBottleneck(
                dim=feat_channels[-1],
                num_heads=attn_heads,
                window_size=4,
                depth=2,
            )
        self.fusion = CrossAttentionFusion(
            feat_dim=feat_channels[-1], embed_dim=embed_dim, num_heads=attn_heads
        )
        self.decoder = DualHeadDecoder(
            encoder_channels=feat_channels,
            embed_dim=embed_dim,
            out_channels=3,
            use_illum_chroma_field=self.use_illum_chroma_field,
        )
        self.sh_head = ProbeSHHead(embed_dim=embed_dim, l_max=sh_l_max)

        # Global illuminant estimation head (training-only aux). Predicts the
        # scene illuminant chromaticity from illum_emb; supervised by
        # losses.illuminant in angular space (training-only color-constancy
        # aux). Output is never consumed downstream, so it is safe on or off.
        if self.use_illuminant_head:
            from .illuminant_head import IlluminantHead

            self.illuminant_head = IlluminantHead(embed_dim=embed_dim)

        # Light-direction head (contribution C). Predicts (φ, θ, b) from the s4
        # bottleneck and routes it through illum_embed at inference, replacing
        # the null token so the deploy path conditions on a learned direction.
        if self.use_direction_head:
            self.direction_head = DirectionHead(feat_dim=feat_channels[-1])

        # P1.3 — specular branch (off by default). Reads the input RGB so it
        # works at inference without exposing decoder internals.
        if self.use_specular_head:
            self.specular_head = SpecularHead(in_channels=3, hidden=32)

        # P2.2 — latent illumination CLS token. Cross-attends to the
        # bottleneck features and adds an inference-time illumination
        # contribution to the embedding (zero-init out_proj keeps identity).
        if self.use_illum_token:
            self.illum_token = LatentIlluminationToken(
                feat_dim=feat_channels[-1],
                embed_dim=embed_dim,
                num_heads=max(1, attn_heads // 2),
            )

        # P2.2 — reflectance memory bank. Not wired into the forward path by
        # default; available for ablations and post-hoc lookups.
        if self.use_reflectance_memory:
            self.reflectance_memory = ReflectanceMemoryBank(
                num_slots=int(reflectance_memory_size), dim=embed_dim
            )

        # Auxiliary material classification head (training-only). Tap point
        # is s4 post-SwinV2 pre-Fusion — see the `forward` body below.
        if self.use_material_head:
            self.material_head = MaterialHead(
                in_channels=feat_channels[-1],
                num_classes=self.material_num_classes,
                hidden=int(material_head_hidden),
            )

    def forward(
        self,
        rgb: torch.Tensor,
        *,
        seg: torch.Tensor | None = None,
        phi: torch.Tensor | None = None,
        theta: torch.Tensor | None = None,
        bnorm: torch.Tensor | None = None,
        direction_id: torch.Tensor | None = None,
        normals: torch.Tensor | None = None,
        compute_material: bool = True,
        use_pred_direction: bool = False,
    ) -> dict[str, torch.Tensor]:
        B = rgb.shape[0]
        use_ckpt = self.activation_checkpoint and self.training and rgb.requires_grad

        seg_emb = (
            self.seg_encoder(seg)
            if (seg is not None and self.use_sam_conditioning)
            else self.null_seg_emb.expand(B, -1)
        )
        # Illumination conditioning is resolved AFTER the encoder/bottleneck so
        # the direction head + illum_token can read the s4 features — see below.

        if use_ckpt:
            from torch.utils.checkpoint import checkpoint

            feats = checkpoint(self.encoder, rgb, seg_emb, normals, use_reentrant=False)
        else:
            feats = self.encoder(rgb, seg_emb, normals=normals)
        s1, s2, s3, s4 = feats[0], feats[1], feats[2], feats[3]

        # SwinV2 bottleneck: residual self-attention on s4 BEFORE the
        # cross-attention fusion writes the illumination residual.
        if self.use_swin_bottleneck:
            if use_ckpt:
                from torch.utils.checkpoint import checkpoint

                s4 = checkpoint(self.swin_bottleneck, s4, use_reentrant=False)
            else:
                s4 = self.swin_bottleneck(s4)

        # Material head tap: post-SwinV2, pre-fusion. Material decisions are
        # made on illumination-invariant features (cross-attention has not
        # yet written direction-specific residuals into s4).
        material_logits = None
        if self.use_material_head and compute_material:
            material_logits = self.material_head(s4, out_hw=rgb.shape[-2:])

        # ---- illumination conditioning (needs s4) ----
        # Continuous needs (φ, θ, b); categorical needs direction_id. A missing
        # signal falls back on the learned null token, EXCEPT when the
        # direction head is on: it predicts (φ, θ, b) from s4 and routes them
        # through the same illum_embed MLP, so inference conditions on a learned
        # direction instead of the null token.
        dir_pred_enc: torch.Tensor | None = None
        if self.direction_encoding == "categorical":
            illum_emb = (
                self.illum_embed(None, None, None, direction_id=direction_id)
                if direction_id is not None
                else self.null_illum_emb.expand(B, -1)
            )
        else:
            gt_available = phi is not None and theta is not None and bnorm is not None
            if self.use_direction_head:
                dir_pred_enc = self.direction_head(s4)  # [B, 5], always (for the loss)
                if gt_available and not use_pred_direction:
                    illum_emb = self.illum_embed(phi, theta, bnorm)  # teacher forcing
                else:
                    illum_emb = self.illum_embed.forward_encoded(dir_pred_enc)  # predicted
            elif gt_available:
                illum_emb = self.illum_embed(phi, theta, bnorm)
            else:
                illum_emb = self.null_illum_emb.expand(B, -1)

        # Latent illumination token: cross-attends to s4 bottleneck. Adds a
        # learned contribution to the illumination embedding so the model is
        # not solely reliant on the (φ, θ, bnorm) MLP at inference.
        if self.use_illum_token:
            illum_emb = illum_emb + self.illum_token(s4)

        s4_fused = self.fusion(s4, illum_emb)

        R, L = self.decoder(
            s4_fused, skips=[s1, s2, s3], illum_emb=illum_emb, x_rgb=rgb
        )
        I_out = torch.clamp(R * L, 0.0, 1.0)
        sh_pred = self.sh_head(illum_emb)

        out = {
            "output": I_out,
            "reflectance": R,
            "illumination": L,
            "sh_pred": sh_pred,
            "illum_emb": illum_emb,
            "seg_emb": seg_emb,
        }
        if self.use_specular_head:
            logit = self.specular_head(rgb)
            out["specular_logit"] = logit
            out["specular_prob"] = torch.sigmoid(logit)
        if material_logits is not None:
            out["material_logits"] = material_logits
        if self.use_illuminant_head:
            out["illuminant"] = self.illuminant_head(illum_emb)
        if dir_pred_enc is not None:
            out["dir_pred_enc"] = dir_pred_enc
        return out
