"""Train-time augmentations for dainet.

All augmentations apply to the **input** image only; the target is the
canonical flat-lit reference and must remain unmodified, so the model
learns to *invert* the perturbations back to the canonical appearance.

Each augmentation is a small callable taking a uint8 RGB array `[H, W, 3]`
and a numpy RNG, returning a uint8 RGB array of the same shape. The
composer applies them in a fixed order with config-controlled probabilities.

Why these specific augmentations:

- **ColorTemperatureJitter** — simulates warm/cool light casts (3500K-8500K).
  This is the single most useful illumination augmentation for the
  ΔE2000<3 target: the model otherwise only sees the natural-cast
  distribution of MIT-MI.
- **ExposureJitter** — random gamma + gain teaches the network to handle
  over- and under-exposure, where the failure analysis sees most clipping.
- **SyntheticSpecularSpot** — pastes Gaussian bright spots; combined with
  highlight masking (in losses/manager.py) the model learns to recover the
  underlying chromaticity.
- **LocalIlluminationGradient** — multiplies the image by a smooth
  low-frequency RGB gradient, simulating mixed lighting (daylight from a
  window + tungsten lamp in the same room).
- **SensorNoise** — Gaussian additive noise, robust low-light.
- **BrightnessJitter** — kept as the chromaticity-preserving baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / 255.0


def _as_u8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)


def _cct_to_rgb_gain(cct_k: float) -> np.ndarray:
    """Approximate per-channel RGB gain to *apply to a D65 image* so it looks
    as if shot under illuminant CCT in Kelvin. Uses Tanner Helland's
    piecewise approximation (good enough for synthetic cast augmentation).
    The returned vector is normalised so the green channel equals 1.
    """
    t = cct_k / 100.0
    # Red
    if t <= 66.0:
        r = 255.0
    else:
        r = 329.698727446 * ((t - 60.0) ** -0.1332047592)
    # Green
    if t <= 66.0:
        g = 99.4708025861 * np.log(max(t, 1.0)) - 161.1195681661
    else:
        g = 288.1221695283 * ((t - 60.0) ** -0.0755148492)
    # Blue
    if t >= 66.0:
        b = 255.0
    elif t <= 19.0:
        b = 0.0
    else:
        b = 138.5177312231 * np.log(max(t - 10.0, 1.0)) - 305.0447927307
    gain = np.array([r, g, b], dtype=np.float32) / 255.0
    gain = gain / max(gain[1], 1e-6)
    return gain


# ----------------------------------------------------------------------
# Augmentation classes
# ----------------------------------------------------------------------


@dataclass
class BrightnessJitter:
    enabled: bool = True
    rng_range: tuple[float, float] = (0.92, 1.08)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled:
            return img
        scale = float(rng.uniform(*self.rng_range))
        return _as_u8(np.clip(_as_float(img) * scale, 0.0, 1.0))


@dataclass
class SensorNoise:
    enabled: bool = True
    sigma_range: tuple[float, float] = (0.002, 0.012)
    p: float = 0.5

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or rng.random() >= self.p:
            return img
        sigma = float(rng.uniform(*self.sigma_range))
        f = _as_float(img)
        f = f + rng.normal(0.0, sigma, size=f.shape).astype(np.float32)
        return _as_u8(np.clip(f, 0.0, 1.0))


@dataclass
class ColorTemperatureJitter:
    enabled: bool = True
    strength: float = 0.15  # blend factor 0=off, 1=full chromatic shift
    p: float = 0.5
    cct_range: tuple[float, float] = (3500.0, 8500.0)

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or rng.random() >= self.p:
            return img
        cct = float(rng.uniform(*self.cct_range))
        gain = _cct_to_rgb_gain(cct)  # per-channel multiplier
        # Blend toward target gain: gain_eff = (1-s)*1 + s*gain
        gain_eff = (1.0 - self.strength) + self.strength * gain
        f = _as_float(img) * gain_eff.reshape(1, 1, 3)
        return _as_u8(np.clip(f, 0.0, 1.0))


@dataclass
class ExposureJitter:
    enabled: bool = True
    gamma_range: tuple[float, float] = (0.8, 1.25)
    gain_range: tuple[float, float] = (0.8, 1.2)
    p: float = 0.4

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or rng.random() >= self.p:
            return img
        gamma = float(rng.uniform(*self.gamma_range))
        gain = float(rng.uniform(*self.gain_range))
        f = _as_float(img)
        f = np.clip(f, 1e-6, 1.0)
        f = (f ** gamma) * gain
        return _as_u8(np.clip(f, 0.0, 1.0))


@dataclass
class SyntheticSpecularSpot:
    enabled: bool = True
    max_spots: int = 3
    p: float = 0.15

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or rng.random() >= self.p:
            return img
        f = _as_float(img)
        H, W, _ = f.shape
        n = int(rng.integers(1, self.max_spots + 1))
        ys = np.arange(H)
        xs = np.arange(W)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        for _ in range(n):
            cy = float(rng.uniform(0, H))
            cx = float(rng.uniform(0, W))
            sigma = float(rng.uniform(0.02, 0.08)) * max(H, W)
            blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2.0 * sigma * sigma)))
            f = f + blob[:, :, None].astype(np.float32) * float(rng.uniform(0.6, 1.0))
        return _as_u8(np.clip(f, 0.0, 1.0))


@dataclass
class LocalIlluminationGradient:
    enabled: bool = True
    strength: float = 0.12
    p: float = 0.15

    def __call__(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        if not self.enabled or rng.random() >= self.p:
            return img
        H, W, _ = img.shape
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        # Random RGB gradient direction
        dirs = rng.normal(0.0, 1.0, size=(3, 2)).astype(np.float32)
        gain_field = np.stack(
            [
                1.0 + self.strength * (dirs[c, 0] * yy + dirs[c, 1] * xx)
                for c in range(3)
            ],
            axis=-1,
        )
        gain_field = np.clip(gain_field, 0.6, 1.4)
        f = _as_float(img) * gain_field
        return _as_u8(np.clip(f, 0.0, 1.0))


# ----------------------------------------------------------------------
# Composer
# ----------------------------------------------------------------------


_NAME_TO_CLASS = {
    "brightness": BrightnessJitter,
    "sensor_noise": SensorNoise,
    "color_temperature": ColorTemperatureJitter,
    "exposure": ExposureJitter,
    "synthetic_specular": SyntheticSpecularSpot,
    "local_illumination": LocalIlluminationGradient,
}

# Order matters: chromatic shifts before brightness, brightness before noise
# (noise should not be re-quantised by later gamma). Spec spots last because
# they should not be re-tinted by the color-temp shift.
_AUG_ORDER = [
    "color_temperature",
    "exposure",
    "local_illumination",
    "brightness",
    "synthetic_specular",
    "sensor_noise",
]


def _build_one(name: str, cfg: dict[str, Any]) -> Any:
    cls = _NAME_TO_CLASS[name]
    kwargs: dict[str, Any] = {}
    if "enabled" in cfg:
        kwargs["enabled"] = bool(cfg["enabled"])
    if name == "brightness" and "range" in cfg:
        kwargs["rng_range"] = tuple(cfg["range"])
    if name == "sensor_noise":
        if "sigma_range" in cfg:
            kwargs["sigma_range"] = tuple(cfg["sigma_range"])
        if "p" in cfg:
            kwargs["p"] = float(cfg["p"])
    if name == "color_temperature":
        if "strength" in cfg:
            kwargs["strength"] = float(cfg["strength"])
        if "p" in cfg:
            kwargs["p"] = float(cfg["p"])
        if "cct_range" in cfg:
            kwargs["cct_range"] = tuple(cfg["cct_range"])
    if name == "exposure":
        if "gamma_range" in cfg:
            kwargs["gamma_range"] = tuple(cfg["gamma_range"])
        if "gain_range" in cfg:
            kwargs["gain_range"] = tuple(cfg["gain_range"])
        if "p" in cfg:
            kwargs["p"] = float(cfg["p"])
    if name == "synthetic_specular":
        if "max_spots" in cfg:
            kwargs["max_spots"] = int(cfg["max_spots"])
        if "p" in cfg:
            kwargs["p"] = float(cfg["p"])
    if name == "local_illumination":
        if "strength" in cfg:
            kwargs["strength"] = float(cfg["strength"])
        if "p" in cfg:
            kwargs["p"] = float(cfg["p"])
    return cls(**kwargs)


class AugmentPipeline:
    """Apply a fixed-order pipeline of augmentations to the input only."""

    def __init__(self, cfg: dict[str, Any] | None):
        cfg = cfg or {}
        self.augs = []
        for name in _AUG_ORDER:
            entry = cfg.get(name, {})
            if not isinstance(entry, dict):
                entry = {"enabled": bool(entry)}
            if entry.get("enabled", True):
                self.augs.append(_build_one(name, entry))

    def __call__(self, img_u8: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out = img_u8
        for aug in self.augs:
            out = aug(out, rng)
        return out
