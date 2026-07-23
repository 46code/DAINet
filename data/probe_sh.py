"""Project a chrome-sphere probe image into real spherical-harmonic coefficients.

A chrome sphere imaged under orthographic projection samples the reflected
direction `r = 2(n·v)n − v` at each pixel inside the silhouette, where the
camera direction is `v = (0, 0, 1)` and `n` is the sphere's surface normal.
Projecting the chrome pixel intensities onto a real-SH basis up to `l_max`
yields a compact, direction-aware summary of the incoming illumination
environment — used as supervision for the network's illumination embedding
head.

The projection uses uniform pixel weights (not solid-angle Jacobian-weighted).
This biases coefficients slightly toward equatorial samples; it is acceptable
for a supervision target because the network's head is asked to mimic the
*same* projection, so the bias cancels.
"""

from __future__ import annotations

import numpy as np
from scipy import special as sp_special


def real_sh_basis(theta: np.ndarray, phi: np.ndarray, l_max: int = 4) -> np.ndarray:
    """Evaluate real SH basis on a direction array.

    Args:
        theta: polar angle in [0, pi], shape [N].
        phi:   azimuth in [0, 2*pi], shape [N].
        l_max: maximum SH order (>= 0).

    Returns:
        sh_vals: [N, (l_max+1)**2] real-valued array. Coefficients are ordered
        l = 0..l_max, m = -l..l.
    """
    n = theta.shape[0]
    n_coeff = (l_max + 1) ** 2
    out = np.empty((n, n_coeff), dtype=np.float64)
    col = 0
    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            cm = sp_special.sph_harm(abs(m), l, phi, theta)
            if m > 0:
                val = np.sqrt(2.0) * ((-1) ** m) * cm.real
            elif m < 0:
                val = np.sqrt(2.0) * ((-1) ** m) * cm.imag
            else:
                val = cm.real
            out[:, col] = val
            col += 1
    return out.astype(np.float32)


def chrome_to_sh(chrome: np.ndarray, l_max: int = 4) -> np.ndarray:
    """Project a chrome sphere image to real-SH coefficients.

    Args:
        chrome: [H, W, 3] float32 in [0, 1] (linear or sRGB; this function does
            not convert, so users can choose). Pass an sRGB image and the
            coefficients describe sRGB intensities.
        l_max: maximum SH order.

    Returns:
        sh: [3, (l_max+1)**2] float32. Per-channel SH coefficients of the
        sampled reflected environment.
    """
    H, W = chrome.shape[:2]
    cy = (H - 1) / 2.0
    cx = (W - 1) / 2.0
    radius = min(H, W) / 2.0 - 1.0
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    # Normalize to sphere coordinates. Image y points down; world y points up.
    u = (xx - cx) / radius
    v = (cy - yy) / radius
    inside = (u * u + v * v) < 1.0

    nx = u[inside]
    ny = v[inside]
    nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))

    # Reflection of view direction (0,0,1) across the surface normal:
    # r = 2(n·v) n − v, with v = (0,0,1).  n·v = nz.
    rx = 2.0 * nz * nx
    ry = 2.0 * nz * ny
    rz = 2.0 * nz * nz - 1.0

    theta = np.arccos(np.clip(rz, -1.0, 1.0)).astype(np.float64)
    phi = np.mod(np.arctan2(ry, rx).astype(np.float64), 2.0 * np.pi)

    basis = real_sh_basis(theta, phi, l_max=l_max)  # [N_pix, n_coeff]

    pixels = chrome[inside].astype(np.float32)  # [N_pix, 3]
    n_pix = basis.shape[0]
    if n_pix == 0:
        return np.zeros((3, (l_max + 1) ** 2), dtype=np.float32)
    # Monte-Carlo estimate of int E(r) Y(r) dω  ≈  (4π / N) Σ_i pixel_i Y_i
    sh = (4.0 * np.pi / n_pix) * (pixels.T @ basis)  # [3, n_coeff]
    # Normalize to unit Frobenius norm so supervision is invariant to exposure
    # / overall chrome brightness — what we want from the SH target is the
    # *shape* of the environment, not its absolute scale.
    norm = float(np.sqrt((sh * sh).sum())) + 1e-8
    sh = sh / norm
    return sh.astype(np.float32)
