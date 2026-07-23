"""Evaluation metrics for dainet — the benchmark umbrella.

Three primary metrics (no secondary tier):
    psnr           — sRGB PSNR
    ms_ssim        — multi-scale SSIM (Wang 2003)
    lpips          — LPIPS-AlexNet (Zhang CVPR 2018)

All metrics accept `pred, target` as [B, 3, H, W] float tensors in [0, 1].
Per-image scalar metrics return [B]; the dataset-level `MetricComputer`
averages them across the loader.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


# --------------------------------------------------------------------------------------
# PSNR
# --------------------------------------------------------------------------------------


def _expand_mask(mask: Optional[torch.Tensor], ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Broadcast a [B,1,H,W] (or [1,H,W]) mask to match ref [B,C,H,W]."""
    if mask is None:
        return None
    if mask.dim() == 3:
        mask = mask.unsqueeze(0)
    return mask.to(device=ref.device, dtype=ref.dtype).expand_as(ref)


def psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    sq = (pred - target) ** 2
    if mask is not None:
        m = _expand_mask(mask, pred)
        denom = m.sum(dim=[1, 2, 3]).clamp_min(1.0)
        mse = (sq * m).sum(dim=[1, 2, 3]) / denom
    else:
        mse = sq.mean(dim=[1, 2, 3])
    return 10.0 * torch.log10(max_val ** 2 / mse.clamp_min(1e-10))


# --------------------------------------------------------------------------------------
# SSIM / MS-SSIM
# --------------------------------------------------------------------------------------


def _gaussian_window(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g1d = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(-1) * g1d.unsqueeze(0)
    return g2d.expand(channels, 1, window_size, window_size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    max_val: float = 1.0,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    C = pred.shape[1]
    win = _gaussian_window(window_size, sigma, C, pred.device, pred.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(pred, win, padding=pad, groups=C)
    mu_y = F.conv2d(target, win, padding=pad, groups=C)
    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y
    sx = F.conv2d(pred * pred, win, padding=pad, groups=C) - mu_x_sq
    sy = F.conv2d(target * target, win, padding=pad, groups=C) - mu_y_sq
    sxy = F.conv2d(pred * target, win, padding=pad, groups=C) - mu_xy
    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2
    num = (2 * mu_xy + c1) * (2 * sxy + c2)
    den = (mu_x_sq + mu_y_sq + c1) * (sx + sy + c2)
    ssim_map = num / den
    if mask is not None:
        m = _expand_mask(mask, pred)
        denom = m.sum(dim=[1, 2, 3]).clamp_min(1.0)
        return (ssim_map * m).sum(dim=[1, 2, 3]) / denom
    return ssim_map.mean(dim=[1, 2, 3])


def ms_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    max_val: float = 1.0,
    weights: Optional[list[float]] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if weights is None:
        weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    msssim: list[torch.Tensor] = []
    x, y = pred, target
    m = _expand_mask(mask, pred) if mask is not None else None
    for i in range(len(weights)):
        msssim.append(ssim(x, y, max_val=max_val, mask=m))
        if i < len(weights) - 1:
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)
            if m is not None:
                m = F.avg_pool2d(m, kernel_size=2)
                # binarize again so partial-overlap windows aren't fractionally weighted
                m = (m > 0.5).to(m.dtype)
    result = torch.ones_like(msssim[0])
    for s, w in zip(msssim, weights):
        result = result * s.clamp_min(1e-6) ** w
    return result


# --------------------------------------------------------------------------------------
# LPIPS
# --------------------------------------------------------------------------------------


def lpips_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    net: str = "alex",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LPIPS via the AlexNet (or VGG) backbone. The mask is applied as a
    multiplicative gate on both inputs so probe pixels share a single value
    in both images and contribute ~zero to the network's perceptual loss.
    """
    from losses.lpips_utils import get_lpips_model

    if mask is not None:
        m = _expand_mask(mask, pred)
        pred = pred * m
        target = target * m
    model = get_lpips_model(net=net).to(pred.device)
    with torch.no_grad():
        scores = model(pred * 2.0 - 1.0, target * 2.0 - 1.0)
    return scores.view(-1)


# --------------------------------------------------------------------------------------
# MetricComputer
# --------------------------------------------------------------------------------------


class MetricComputer:
    """Stateful accumulator over an evaluation loop. Returns averages on .compute().

    The three metrics are all primary — no secondary tier. They form a single
    benchmark umbrella that any reader of a dainet run can compare against
    published baselines directly.
    """

    METRIC_KEYS = (
        "psnr",
        "ms_ssim",
        "lpips",
    )

    def __init__(self, with_lpips: bool = True):
        self.with_lpips = with_lpips
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {k: 0.0 for k in self.METRIC_KEYS}
        self._count: int = 0

    @torch.no_grad()
    def update(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        segments: Optional[torch.Tensor] = None,  # accepted for back-compat; unused
        mask: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Accumulate one batch of metrics. When ``mask`` is provided it must
        be a [B,1,H,W] (or [1,H,W]) tensor with 1=valid scene, 0=excluded
        (e.g. chrome+gray probe regions). Every metric honors the mask.

        ``segments`` is accepted but ignored — the seg-variance metric was
        retired from the benchmark umbrella.
        """
        del segments  # unused; kept in signature for caller back-compat
        pred = pred.detach().float().clamp(0, 1)
        target = target.detach().float().clamp(0, 1)
        if mask is not None:
            mask = mask.detach().float()
        row: dict[str, float] = {}
        row["psnr"] = psnr(pred, target, mask=mask).mean().item()
        row["ms_ssim"] = ms_ssim(pred, target, mask=mask).mean().item()
        if self.with_lpips:
            row["lpips"] = lpips_score(pred, target, mask=mask).mean().item()
        else:
            row["lpips"] = 0.0
        for k, v in row.items():
            self._sums[k] += v * pred.shape[0]
        self._count += pred.shape[0]
        return row

    def compute(self) -> dict[str, float]:
        if self._count == 0:
            return {k: 0.0 for k in self.METRIC_KEYS}
        return {k: self._sums[k] / self._count for k in self.METRIC_KEYS}
