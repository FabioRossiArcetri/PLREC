"""How the multi-wavelength unwrap depends on the band, the piston range and the
per-wavelength error -- with synthetic wrapped estimates, no lantern simulation.

The per-wavelength models give each piston modulo lambda_w with an error whose
RMSE is a fixed fraction f of lambda_w (measured: 1.16% for pl19_800 over
700-900 nm, 1.29% for pl19_1200 over 1100-1300 nm). Here the wrapped estimates
are the true wrapped pistons plus Student-t errors (heavy tails, as the real
residuals) scaled to that RMSE, fused by ml_reconstruct.unwrap.unwrap exactly as
in PL_EML.ipynb. Output: RMSE and wrong-fringe rate per configuration.

    python scripts/unwrap_band_study.py [n_samples]
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from ml_reconstruct.unwrap import unwrap, wrap          # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
N_WL = 21                                                 # per-wavelength models used (PL_EML)


def run(lam_c, half_band, span_waves, f, df, seed=0):
    lam = np.linspace(lam_c - half_band, lam_c + half_band, N_WL)
    g = np.random.default_rng(seed)
    span = span_waves * lam_c
    p = g.uniform(-span, span, N)
    t = g.standard_t(df, (N, N_WL)) if np.isfinite(df) else g.normal(size=(N, N_WL))
    t /= np.sqrt(df / (df - 2)) if np.isfinite(df) else 1.0      # unit variance
    sig = f * lam
    ghat = wrap(p[:, None] + t * sig[None, :], lam[None, :])
    ph, _ = unwrap(ghat, np.ones_like(ghat), lam, sig, span)
    e = ph - p
    wrong = np.abs(e) > lam.min() / 2
    rmse = np.sqrt(np.mean(e ** 2))
    bulk = np.sqrt(np.mean(e[~wrong] ** 2)) if (~wrong).any() else np.nan
    return rmse, 100 * wrong.mean(), bulk


if __name__ == "__main__":
    print(f"{N} samples per row, {N_WL} wavelengths, Student-t errors")
    print(f"{'config':44s} {'df':>4s} {'RMSE':>8s} {'wrong %':>8s} {'bulk':>6s}")
    rows = [
        # calibration against the measured campaigns
        ("pl19_800 old: 700-900, +-6.25 lam, f 1.16%", 800, 100, 6.25, 0.0116),
        ("pl19_1200 now: 1100-1300, +-8 lam, f 1.29%", 1200, 100, 8.0, 0.0129),
        # what changes the 1200 nm result
        ("1200: band +-150 (same 25% as 800)", 1200, 150, 8.0, 0.0129),
        ("1200: band +-200", 1200, 200, 8.0, 0.0129),
        ("1200: +-6.25 lam (old range in waves)", 1200, 100, 6.25, 0.0129),
        ("1200: f 1.0% (better per-lambda models)", 1200, 100, 8.0, 0.010),
        ("800 at the new range: +-8 lam", 800, 100, 8.0, 0.0116),
    ]
    for df in (3.0, 4.0, 6.0):
        for name, lc, hb, sw, f in rows:
            r, w, b = run(lc, hb, sw, f, df)
            print(f"{name:44s} {df:4.0f} {r:8.2f} {w:8.3f} {b:6.2f}", flush=True)
        print()
