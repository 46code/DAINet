<h1 align="center">DAINet — Direction-Aware Illumination Network</h1>

<p align="center">
  <b>Direction-aware, physics-guided single-image illumination normalization for indoor scenes.</b>
</p>

<p align="center">
  <b>Anirban Das</b> &nbsp;·&nbsp; Erasmus Mundus MSc <b>COSI</b> &nbsp;·&nbsp; 2026<br>
  <sub>
  Academic supervisors: <b>Prof. Seyed Ali Amirshahi</b> · <b>Prof. Luis Gómez Robledo</b> &nbsp;|&nbsp;
  Host supervisors: <b>Prof. Theo Gevers</b> · <b>Dr. Sezer Karaoglu</b> (3DUniversum / UvA)<br>
  📧 <a href="mailto:anirbandas.das98@gmail.com">anirbandas.das98@gmail.com</a> &nbsp;·&nbsp; 🐙 <a href="https://github.com/46code">@46code</a>
  </sub>
</p>

<p align="center">
  <img src="assets/readme/before_after.gif" width="60%" alt="Before/after crossfade"><br>
  <sub>Input → DAINet correction (crossfade)</sub>
</p>

---

## Overview

DAINet factorizes a single directional sRGB photo into intrinsic reflectance $R$ and illumination $L$ to reconstruct a flat-lit reference image:

$$I_{\text{out}} = \text{clamp}(R \cdot L, 0, 1)$$

- **Core Finding:** Training on multi-direction lighting diversity provides a **+1.47 dB** gain over single-direction training.
- **Key Advantage:** Achieves state-of-the-art illumination normalization on the MIT-MI benchmark at **126 M** parameters using single-image RGB-only inference.

---

## Benchmark Results (MIT-MI Test)

Models trained from scratch on MIT-MI at an equal budget (30 unseen scenes × 25 directions = 750 images, probe-masked):

| Model | PSNR ↑ | MS-SSIM ↑ | LPIPS ↓ | Params |
| :--- | :---: | :---: | :---: | :---: |
| Restormer (CVPR'22) | 21.10 | 0.846 | 0.165 | — |
| Retinexformer (ICCV'23) | 21.57 | 0.858 | 0.155 | — |
| IFBlend (ECCV'24) | 22.28 | 0.877 | 0.155 | 383 M |
| RLN2 (ICCV'25) | 22.32 | 0.877 | **0.145** | 370 M |
| **DAINet (Ours)** | **23.83** | **0.880** | 0.218 | **126 M** |

---

## Architecture Summary

- **Encoder:** ConvNeXt-Base backbone conditioned via per-stage FiLM with SAM2 segmentation embeddings and DSINE surface-normal residuals.
- **Bottleneck:** Dual SwinV2 self-attention blocks for global context.
- **Direction Conditioning:** MLP encoding flash direction $(\phi, \theta, \log b)$ for cross-attention and FiLM; includes an implicit **DirectionHead** predicting parameters at runtime for metadata-free RGB inference.
- **Decoder:** Five upsampling stages with dual zero-initialized heads outputting $R \in (0, 1)$ and $L \in [0.082, 12.2]$.

---

## Repository Layout

```text
models/        DAINet network architecture, heads, encoders, decoder
losses/        Loss modules & DAINetLoss orchestrator
data/          DAINetDataset, SAM fusion, segmentation, probe handling
training/      Trainer, evaluator, EMA, callbacks
evaluation/    Evaluation metrics (PSNR, MS-SSIM, LPIPS)
configs/       dainet.yaml run config
scripts/       Train, test, infer, and data-preprocessing scripts
