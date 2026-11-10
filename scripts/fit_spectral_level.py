#!/usr/bin/env python
"""Fit the per-wavelength sparse cubic (OMP) of the dispersed lantern on one SCAO-residual level.

The same fit as scripts/score_test.py / wfe_unwrap.ipynb make on spectral_ao120_21wl.npz
(ml.fit_wrapped_models, model "omp", K = k_wrap, screen 2000, all 8000 training rows, inputs
standardised per (wavelength, core) on the training rows), on spectral_ao<rms>_21wl.npz, cached
in cache_ao<rms>_perlambda under the tag omp-K<K> (one pickle per wavelength, so that wavelengths
can be fitted by separate processes). It then scores the held-out rows (8000-10000) and, if it
exists, the independent test set, with the multi-wavelength unwrap, once every wavelength is there.

    CBEAM_LANTERN=pl19_1200 python scripts/fit_spectral_level.py 213 --wl 0 1 2     # fit some
    CBEAM_LANTERN=pl19_1200 python scripts/fit_spectral_level.py 213 --score         # all + score
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import ml_reconstruct as ml                                            # noqa: E402
from ml_reconstruct.presets import settings                            # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("rms", type=float)
ap.add_argument("--wl", type=int, nargs="*", default=None, help="wavelength indices (default all)")
ap.add_argument("--score", action="store_true")
args = ap.parse_args()

S = settings()
DATA = S["data_dir"]
tr = ml.LanternDataset.load(f"{DATA}/spectral_ao{args.rms:g}_21wl.npz")
lam = np.asarray(tr.wavelengths_nm, float)
SB_tr, CB_tr = tr.train
SB_va, CB_va = tr.val
n_tr = len(SB_tr)
xs3 = ml.Standardizer().fit(SB_tr.reshape(n_tr, -1))
X3 = lambda A: xs3.transform(A.reshape(len(A), -1)).reshape(A.shape)
X3_tr, X3_va = X3(SB_tr), X3(SB_va)
K = S["k_wrap"]
cache = ml.ModelCache(f"{DATA}/cache_ao{args.rms:g}_perlambda", verbose=True)
wl = np.arange(len(lam)) if args.wl is None else np.asarray(args.wl)
fit = ml.fit_wrapped_models(X3_tr, X3_va, CB_tr, CB_va, lam, wl, np.arange(SB_tr.shape[2]), degree=3,
                            K=K, screen=2000, cache=cache, tag=f"omp-K{K}", model="omp")
print("wrapped RMSE per wavelength (held-out, nm):", np.round(fit["sig_va"].mean(axis=1), 1))
if args.score:
    span = S["spectral_span_nm"]
    P, _ = ml.unwrap_all(fit, span)
    print(f"held-out rows, unwrapped: RMSE {ml.rmse_nm(P, CB_va).mean():.1f} nm, "
          f"wrong fringes {100 * np.mean(np.abs(P - CB_va) > lam.min() / 2):.2f} %")
    test = f"{DATA}/spectral_ao{args.rms:g}_21wl_test.npz"
    if os.path.exists(test):
        te = ml.LanternDataset.load(test)
        ft = ml.fit_wrapped_models(X3_tr, X3(te.signals), CB_tr, te.coeff[:, :5], lam, wl,
                                   np.arange(SB_tr.shape[2]), degree=3, K=K, screen=2000, cache=cache,
                                   tag=f"omp-K{K}", model="omp")
        P, _ = ml.unwrap_all(ft, span)
        C = te.coeff[:, :5]
        print(f"test set, unwrapped: RMSE {ml.rmse_nm(P, C).mean():.1f} nm, "
              f"wrong fringes {100 * np.mean(np.abs(P - C) > lam.min() / 2):.2f} %; "
              f"per-wavelength wrapped {ft['sig_va'].mean():.1f} nm")
