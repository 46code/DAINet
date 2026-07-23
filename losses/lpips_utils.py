"""Lazy LPIPS model loader.

`lpips.LPIPS` is heavy to import and instantiate; this module defers loading
until the first call and caches the model in a module-level dict keyed by
network name. Default backbone is AlexNet — the literature-standard backbone
for low-level vision benchmarks (Zhang et al. CVPR 2018) and what virtually
every WB / color-constancy / low-light paper reports. Training loss and the
val/test metric both consume the same AlexNet model so the two are directly
comparable.
"""

from __future__ import annotations

import warnings

import torch

_LPIPS_MODELS: dict[str, "torch.nn.Module"] = {}


def get_lpips_model(net: str = "alex"):
    if net in _LPIPS_MODELS:
        return _LPIPS_MODELS[net]
    # The lpips package (and its bundled AlexNet/VGG loaders) still call
    # torchvision with the deprecated `pretrained=True` kwarg and
    # `torch.load(..., weights_only=False)`. Both produce noisy
    # UserWarning / FutureWarning at instantiation. Suppress at the
    # boundary — we don't control the upstream code path.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*parameter 'pretrained' is deprecated.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*Arguments other than a weight enum or `None` for 'weights'.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=".*`weights_only=False`.*",
            category=FutureWarning,
        )
        import lpips  # heavy import — pay it once

        model = lpips.LPIPS(net=net, verbose=False).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _LPIPS_MODELS[net] = model
    return model


def lpips_loss(
    pred: torch.Tensor, target: torch.Tensor, net: str = "alex"
) -> torch.Tensor:
    """LPIPS perceptual distance. Inputs in [0, 1] are rescaled to [-1, 1]."""
    model = get_lpips_model(net=net).to(pred.device)
    return model(pred * 2.0 - 1.0, target * 2.0 - 1.0).mean()
