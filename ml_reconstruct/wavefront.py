"""Wavefront errors outside the sensed petal pistons: f^-alpha phase screens.

The ANDES error budget models the optical surface errors and the non-common
path aberrations (NCPA) inside the instrument with a power spectral density
PSD(f) ~ f^-2 (30 nm rms allocated to optical surfaces; the NCPA split into
~11.9 nm rms of low-order modes, up to spherical aberration, and ~27.5 nm rms
beyond). ``psd_screens`` draws such screens on the pupil grid, removes their
global-piston and petal-piston content (the pistons are the quantity to sense:
the truth stays the DM piston coefficients) and scales each to an exact rms.
``low_order_split`` measures how a set of screens divides between Zernike
modes up to spherical aberration (Noll 2..11) and the rest.
"""
from __future__ import annotations

import numpy as np


def psd_screens(n, mask, rms_nm, alpha=2.0, rng=0, remove=None, pad=2, f_max=None):
    """``n`` OPD screens [nm] on the pupil-mask pixels (row-major order of
    ``mask``, the order of the DM influence functions), PSD ~ f^-alpha.

    mask   : (N, N) bool pupil.
    f_max  : highest spatial frequency [cycles per pupil diameter]; None: up
             to the grid's Nyquist frequency. With f^-2 the low-order share of
             the variance depends on it (``low_order_split``).
    remove : (k, n_pix) maps whose span is projected out of every screen
             (e.g. the petal-piston influence functions); a constant (global
             piston) is always removed.
    pad    : the screens are drawn on a (pad*N)^2 periodic grid and cropped,
             so the pupil sees frequencies down to 1/(pad*D).
    Each screen is scaled to exactly ``rms_nm`` rms over the pupil pixels.
    Returns (n, n_pix).
    """
    rng = np.random.default_rng(rng)
    mask = np.asarray(mask, bool)
    N = mask.shape[0]
    M = pad * N
    fx = np.fft.fftfreq(M)
    f = np.hypot(*np.meshgrid(fx, fx, indexing="ij"))
    amp = np.zeros_like(f)
    amp[f > 0] = f[f > 0] ** (-alpha / 2)                 # |FT| ~ sqrt(PSD)
    if f_max is not None:
        ys, xs = np.nonzero(mask)
        diameter_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        amp[f > f_max / diameter_px] = 0.0                # f in cycles per pixel
    B = np.ones((1, int(mask.sum())))
    if remove is not None:
        B = np.vstack([B, np.asarray(remove, float)])
    Q, _ = np.linalg.qr(B.T)                               # orthonormal basis of the removed span
    out = np.empty((n, int(mask.sum())))
    for i in range(n):
        noise = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
        s = np.fft.ifft2(noise * amp).real[:N, :N][mask]
        s -= Q @ (Q.T @ s)
        out[i] = s * (rms_nm / np.sqrt(np.mean(s ** 2)))
    return out


def low_order_split(screens, mask, j_max=11, remove_piston=True):
    """rms [nm] of the part of the screens in Noll Zernikes 2..j_max (up to
    spherical aberration for j_max = 11) and of the rest, least-squares fitted
    on the pupil pixels; mean over the screens. Returns (low, high)."""
    from .zernike import zernike
    mask = np.asarray(mask, bool)
    N = mask.shape[0]
    y, x = (np.mgrid[:N, :N] - (N - 1) / 2) / (N / 2)
    r, t = np.hypot(x, y)[mask], np.arctan2(y, x)[mask]
    Z = np.stack([zernike(j, r, t) for j in range(1, j_max + 1)], axis=1)
    if remove_piston:
        Z = Z[:, 1:]
    coef, *_ = np.linalg.lstsq(Z, np.asarray(screens).T, rcond=None)
    low = Z @ coef
    high = np.asarray(screens).T - low
    return float(np.sqrt(np.mean(low ** 2))), float(np.sqrt(np.mean(high ** 2)))
