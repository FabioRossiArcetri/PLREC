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


def ao_residual_psd(f, n_act=79, inband_share=0.41, L0_over_D=25.0 / 39.0):
    """Radial shape (arbitrary units) of an SCAO residual phase PSD, f in cycles per pupil
    diameter; the variance split is applied in ``ao_residual_screens``.

    f >= f_c = n_act / 2  (beyond the DM control frequency, the higher modes): uncorrected
        turbulence, the fitting error -- von Karman, (f^2 + (D/L0)^-2)^(-11/6).
    f <  f_c (the corrected modes): the residual of the loop. Servo-lag dominated: the
        turbulence spectrum times (f/f_c)^2, i.e. ~ f^(-5/3); noise and aliasing are folded
        into the same shape (a simplification).
    Returns (psd_in, psd_out): the two components, each zero outside its domain."""
    f = np.asarray(f, float)
    fc = n_act / 2.0
    vk = (f ** 2 + (1.0 / L0_over_D) ** 2) ** (-11.0 / 6.0)
    psd_in = np.where((f > 0) & (f < fc), vk * (f / fc) ** 2, 0.0)
    psd_out = np.where(f >= fc, vk, 0.0)
    return psd_in, psd_out


def ao_residual_screens(n, mask, rms_nm, n_act=79, inband_share=0.41, rng=0, remove=None, pad=2,
                        L0_over_D=25.0 / 39.0):
    """``n`` OPD screens [nm] of an SCAO residual (ao_residual_psd): the corrected band
    (f < n_act/2 cycles per pupil) holds ``inband_share`` of the variance, the uncorrected
    fitting error beyond it the rest (default 0.41: the median MICADO SCAO budget, 88 nm
    of bandwidth + noise + aliasing vs 106 nm of fitting error). Piston and the span of
    ``remove`` projected out; each screen scaled to exactly ``rms_nm`` rms over the pupil.
    Frequencies above the grid's Nyquist (pupil pixels / 2 cycles per pupil) are not
    represented. Returns (n, n_pix)."""
    rng = np.random.default_rng(rng)
    mask = np.asarray(mask, bool)
    N = mask.shape[0]
    M = pad * N
    ys, xs = np.nonzero(mask)
    diameter_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    fx = np.fft.fftfreq(M) * diameter_px                  # cycles per pupil diameter
    f = np.hypot(*np.meshgrid(fx, fx, indexing="ij"))
    p_in, p_out = ao_residual_psd(f, n_act, inband_share, L0_over_D)
    # weights so that the two bands carry inband_share and 1 - inband_share of the variance
    w_in = np.sqrt(inband_share / p_in.sum()) if p_in.sum() > 0 else 0.0
    w_out = np.sqrt((1 - inband_share) / p_out.sum()) if p_out.sum() > 0 else 0.0
    amp = w_in * np.sqrt(p_in) + w_out * np.sqrt(p_out)
    B = np.ones((1, int(mask.sum())))
    if remove is not None:
        B = np.vstack([B, np.asarray(remove, float)])
    Q, _ = np.linalg.qr(B.T)
    out = np.empty((n, int(mask.sum())))
    for i in range(n):
        noise = rng.standard_normal((M, M)) + 1j * rng.standard_normal((M, M))
        s = np.fft.ifft2(noise * amp).real[:N, :N][mask]
        s -= Q @ (Q.T @ s)
        out[i] = s * (rms_nm / np.sqrt(np.mean(s ** 2)))
    return out


def band_split(screens, mask, fc, pad=2):
    """rms [nm] of the screens below and above fc cycles per pupil (FFT of the pupil-filled
    screens), mean over the screens. Returns (inband, outband)."""
    mask = np.asarray(mask, bool)
    N = mask.shape[0]
    M = pad * N
    ys, xs = np.nonzero(mask)
    diameter_px = max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
    fx = np.fft.fftfreq(M) * diameter_px
    f = np.hypot(*np.meshgrid(fx, fx, indexing="ij"))
    tot_in = tot_out = 0.0
    for s in screens:
        g = np.zeros((M, M)); g[:N, :N][mask] = s
        P = np.abs(np.fft.fft2(g)) ** 2
        tot_in += P[f < fc].sum(); tot_out += P[f >= fc].sum()
    norm = (tot_in + tot_out) / np.mean([np.mean(s ** 2) for s in screens])
    return float(np.sqrt(tot_in / norm)), float(np.sqrt(tot_out / norm))


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
