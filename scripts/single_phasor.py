#!/usr/bin/env python
"""Single-wavelength reconstructors with phasor targets, and where the gap to
the per-wavelength models of the spectral pipeline comes from.

At one wavelength the signals depend on each piston only modulo lambda. The
single-wavelength reconstructors regress p itself, a target with a jump at
+-lambda/2; the per-wavelength models of the spectral pipeline regress
(sin, cos)(2 pi p / lambda), which has none, and reach ~15 nm wrapped error
where the best single-lambda model is at ~114 nm modulo lambda. Three things
differ between the two; this script changes them one at a time:

    target   p  vs  phasor            (single dataset, 8000 training samples)
    signals  camera windows  vs  channel powers at the design lambda
    samples  8000  vs  40000          (spectral dataset at the design lambda)

Every model is scored by the RMSE of its errors modulo lambda (rmse_nm with
period). Fits are cached in <data_dir>/cache_single_phasor.

    CBEAM_LANTERN=pl19_1200 python scripts/single_phasor.py [--no-mlp]
"""
import os
import sys
import time

os.environ.setdefault("CBEAM_BACKEND", "numpy")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np                                                    # noqa: E402
import ml_reconstruct as ml                                           # noqa: E402
from ml_reconstruct.presets import settings                           # noqa: E402
from ml_reconstruct.unwrap import phasor_targets, phasor_decode       # noqa: E402

S = settings()
L, DATA = S["lantern"], S["data_dir"]
LAM = 2 * S["single_span_nm"]                   # the single range is +-lambda/2
K = S["omp_record"][0]                          # the full cubic budget (1282 / 119)
SCREEN = 9000
MLP = "--no-mlp" not in sys.argv
cache = ml.ModelCache(f"{DATA}/cache_single_phasor")
rows = []


def report(name, C_hat, C_true, n_train, dof=None):
    r = ml.rmse_nm(C_hat, C_true, LAM)
    rows.append((name, n_train, r.mean(), dof))
    print(f"{name:44s} n_train {n_train:6d}   mod-lambda RMSE {r.mean():7.1f} nm   per piston {np.round(r, 1)}",
          flush=True)


def omp_pair(tag, X_tr, X_va, C_tr, C_va, n_train):
    """Cubic OMP with K terms: direct p target and phasor target."""
    ys = ml.Standardizer().fit(C_tr)
    direct = cache(f"{tag}-direct-K{K}", lambda: ml.fit_omp_poly(X_tr, ys.transform(C_tr), 3, K, SCREEN))
    report(f"{tag}: OMP cubic K={K}, target p", ys.inverse(ml.predict_omp_poly(direct, X_va)), C_va, n_train)
    phasor = cache(f"{tag}-phasor-K{K}", lambda: ml.fit_omp_poly(X_tr, phasor_targets(C_tr, LAM), 3, K, SCREEN))
    report(f"{tag}: OMP cubic K={K}, target phasor", phasor_decode(ml.predict_omp_poly(phasor, X_va), LAM)[0],
           C_va, n_train)


def mlp_phasor(tag, X_tr, X_va, C_tr, C_va, n_train):
    """19 -> 3x128 -> 10 (sin, cos per piston), the single-lambda recipe otherwise."""
    layers = [X_tr.shape[1], 128, 128, 128, 2 * C_tr.shape[1]]
    path = f"{DATA}/cache_single_phasor/{tag}-mlp-{'x'.join(map(str, layers))}-e500.npz"
    if os.path.exists(path):
        params, _, _ = ml.load_mlp(path)
    else:
        t = time.time()
        params, hist = ml.train_mlp(layers, X_tr, phasor_targets(C_tr, LAM), X_va, phasor_targets(C_va, LAM),
                                    epochs=500, learning_rate=3e-4)
        ml.save_mlp(path, params, layers, train=hist["train"], val=hist["val"], best_epoch=hist["best_epoch"])
        print(f"   trained in {time.time() - t:.0f} s, best epoch {hist['best_epoch']}")
    report(f"{tag}: MLP 3x128, target phasor", phasor_decode(np.asarray(ml.mlp_forward(params, X_va)), LAM)[0],
           C_va, n_train)


print(f"lantern {L}, lambda {LAM:.0f} nm, K {K}")
# --- 1. the single dataset (camera windows, 8000 training samples): target p vs phasor
ds = ml.LanternDataset.load(S["single"])
S_tr, C_tr = ds.train
S_va, C_va = ds.val
xs = ml.Standardizer().fit(S_tr)
X_tr, X_va = xs.transform(S_tr), xs.transform(S_va)
omp_pair("single", X_tr, X_va, C_tr, C_va, len(X_tr))
if MLP:
    mlp_phasor("single", X_tr, X_va, C_tr, C_va, len(X_tr))

# --- 2. the spectral dataset at the design wavelength (channel powers, pistons over +-8 lambda,
#        wrapped): the same models on 8000 and on all 40000 training samples
sp = ml.LanternDataset.load(S["spectral"])
w = int(np.argmin(np.abs(np.asarray(sp.wavelengths_nm) - LAM)))
P_tr, Cs_tr = sp.train
P_va, Cs_va = sp.val
P_tr, P_va = P_tr[:, w, :], P_va[:, w, :]
Cs_tr, Cs_va = ml.wrap(Cs_tr, LAM), ml.wrap(Cs_va, LAM)          # the piston modulo lambda
print(f"spectral dataset at {sp.wavelengths_nm[w]:.0f} nm (index {w})")
for n in (len(X_tr), len(P_tr)):
    xs2 = ml.Standardizer().fit(P_tr[:n])
    Z_tr, Z_va = xs2.transform(P_tr[:n]), xs2.transform(P_va)
    omp_pair(f"channel-n{n}", Z_tr, Z_va, Cs_tr[:n], Cs_va, n)

print("\nsummary (mod-lambda RMSE, nm)")
for name, n, r, _ in rows:
    print(f"  {name:44s} {n:6d}  {r:7.1f}")
