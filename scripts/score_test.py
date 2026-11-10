#!/usr/bin/env python
"""Score every saved reconstructor on the independent test sets (gen_wfe*.py with WFE_TEST=1).

The validation rows choose things (the MLPs' best epoch, the OMP budget K); the test sets --
2000 new piston configurations and new SCAO-residual screens -- choose nothing, so their RMSE
is unbiased. No model is refitted: the fits are loaded from the caches written by
trainMLP.ipynb (cache_single_phasor, tags ao<rms>-*), wfe_unwrap.ipynb (cache_ao120_perlambda)
and the 4x256 MLPs (cache_big). Symbolic regression is not re-scored (its expressions are
stored with their predictions only; its choice was made on held-out training rows).

    CBEAM_LANTERN=pl19_1200 python scripts/score_test.py        -> <data_dir>/test_scores.npz + table
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import ml_reconstruct as ml                                            # noqa: E402
from ml_reconstruct import forward as fw, petals                       # noqa: E402
from ml_reconstruct.presets import settings                            # noqa: E402
from ml_reconstruct.simulate import bp                                 # noqa: E402

S = settings()
L, DATA = S["lantern"], S["data_dir"]
LAM = 2 * S["single_span_nm"]
LEVELS = [0, 30, 60, 90, 120, 150]
KS = S["wfe_ks"]
cache = ml.ModelCache(f"{DATA}/cache_single_phasor", verbose=False)
_p = bp.get_simulation_parameters(lantern=L)
_f = bp.load_ifunc(_p["ifunc_file"]); bp.center_ifunc_mask(_f)
_g = petals.petal_geometry(np.asarray(_f.mask_inf_func) > 0, np.asarray(_f.influence_function), 5)
TWIN = petals.twin_matrix(_g["opposite"], _g["ref"])
ys = ml.PhasorCodec(LAM)
MLP_SMALL = "mlp-19x128x128x128x10-e500.npz"


def predict_level(tag, S_tr, X_tr, X, S_raw, xs):
    """{method: pistons [nm]} of the models saved under tag, on standardised inputs X."""
    F = ml.poly_features(X_tr, X, 3)[1]
    out = {"linear": ys.inverse(cache(f"{tag}-linear", None).predict(X)),
           "dense poly, degree 2": ys.inverse(ml.predict_dense_poly_ridge(cache(f"{tag}-d2", None), X)),
           "dense poly, degree 3": ys.inverse(ml.predict_dense_poly_ridge(cache(f"{tag}-d3", None), X)),
           "closed form": fw.invert_fixed(cache(f"{tag}-closed-form", None), S_raw, TWIN)[0]}
    for K in KS:
        fits = cache(f"{tag}-omp-K{K}", None)
        out[f"omp K={K}"] = ys.inverse(np.stack([ml.omp_predict(F, f) for f in fits], axis=1))
    out["MLP 3x128"] = ys.inverse(np.asarray(ml.mlp_forward(ml.load_mlp(f"{DATA}/cache_single_phasor/{tag}-{MLP_SMALL}")[0], X)))
    big = f"{DATA}/cache_big/{tag}-mlp-256x4.npz"
    if os.path.exists(big):
        out["MLP 4x256"] = ys.inverse(np.asarray(ml.mlp_forward(ml.load_mlp(big)[0], X)))
    return out


single = {}                       # method -> {level: (validation, test) RMSE modulo lambda}
for r in LEVELS:
    tr = ml.LanternDataset.load(f"{DATA}/single_ao{r}.npz")
    te = ml.LanternDataset.load(f"{DATA}/single_ao{r}_test.npz")
    S_tr, _ = tr.train
    S_va, C_va = tr.val
    S_te, C_te = te.signals, te.coeff[:, :5]
    xs = ml.Standardizer().fit(S_tr)
    X_tr = xs.transform(S_tr)
    pv = predict_level(f"ao{r}", S_tr, X_tr, xs.transform(S_va), S_va, xs)
    pt = predict_level(f"ao{r}", S_tr, X_tr, xs.transform(S_te), S_te, xs)
    for m in pv:
        single.setdefault(m, {})[r] = (ml.rmse_nm(pv[m], C_va, LAM).mean(), ml.rmse_nm(pt[m], C_te, LAM).mean())
    # the OMP budget chosen on the validation rows, reported on the test rows
    kbest = min(KS, key=lambda K: single[f"omp K={K}"][r][0])
    single.setdefault("sparse poly (OMP), K by validation", {})[r] = single[f"omp K={kbest}"][r]
    if r:                         # the clean-trained cubic OMP on this level
        t0 = ml.LanternDataset.load(f"{DATA}/single_ao0.npz")
        x0 = ml.Standardizer().fit(t0.train[0])
        F0 = ml.poly_features(x0.transform(t0.train[0]), x0.transform(S_te), 3)[1]
        fits = cache(f"ao0-omp-K{max(KS)}", None)
        c0 = ys.inverse(np.stack([ml.omp_predict(F0, f) for f in fits], axis=1))
        single.setdefault("OMP trained clean", {})[r] = (np.nan, ml.rmse_nm(c0, C_te, LAM).mean())
    print(f"level {r} done", flush=True)

print("\nsingle wavelength, RMSE modulo lambda (nm): validation / test")
print(f"{'method':38s}" + "".join(f"{r:>16g}" for r in LEVELS))
for m, d in single.items():
    print(f"{m:38s}" + "".join(f"{d[r][0]:7.2f} /{d[r][1]:7.2f}" if r in d else f"{'-':>16s}" for r in LEVELS))

# --------------------------------------------------------------- spectral, 120 nm, each method + unwrap
tr = ml.LanternDataset.load(f"{DATA}/spectral_ao120_21wl.npz")
te = ml.LanternDataset.load(f"{DATA}/spectral_ao120_21wl_test.npz")
lam = np.asarray(tr.wavelengths_nm, float)
SB_tr, CB_tr = tr.train
SB_va, CB_va = tr.val
SB_te, CB_te = te.signals, te.coeff[:, :5]
n_tr = len(SB_tr)
xs3 = ml.Standardizer().fit(SB_tr.reshape(n_tr, -1))
X3 = lambda A: xs3.transform(A.reshape(len(A), -1)).reshape(A.shape)
X3_tr, X3_va, X3_te = X3(SB_tr), X3(SB_va), X3(SB_te)
SPAN = S["spectral_span_nm"]
wcache = ml.ModelCache(f"{DATA}/cache_ao120_perlambda", verbose=False)
METHODS = [("linear", "ridge", 1), ("dense poly, degree 2", "ridge", 2), ("dense poly, degree 3", "ridge", 3),
           ("sparse poly (OMP)", "omp", 3), ("MLP 3x128", "mlp", None), ("closed form", "closed_form", None)]
spectral = {}
for name, kind, deg in METHODS:
    tag = f"{kind}{'-d%d' % deg if deg and kind == 'ridge' else ''}{'-K%d' % S['k_wrap'] if kind == 'omp' else ''}"
    res = []
    for X_e, C_e in ((X3_va, CB_va), (X3_te, CB_te)):
        fit = ml.fit_wrapped_models(X3_tr, X_e, CB_tr, C_e, lam, np.arange(len(lam)), np.arange(19), degree=deg or 3,
                                    K=S["k_wrap"], screen=2000, cache=wcache, tag=tag, model=kind, twin=TWIN)
        P, _ = ml.unwrap_all(fit, SPAN)
        res.append((ml.rmse_nm(P, C_e).mean(), 100 * np.mean(np.abs(P - C_e) > lam.min() / 2)))
    spectral[name] = res
    print(f"  {name}: done", flush=True)
# the 4x256 per-wavelength MLPs: wrapped estimates, training-row errors as weights
gh = {"va": np.zeros((len(CB_va), 5, len(lam))), "te": np.zeros((len(CB_te), 5, len(lam)))}
cf = {k: np.zeros_like(v) for k, v in gh.items()}
sig_tr = np.zeros((len(lam), 5))
for w, lw in enumerate(lam):
    xw = ml.Standardizer().fit(SB_tr[:, w, :])
    cw = ml.PhasorCodec(lw)
    params = ml.load_mlp(f"{DATA}/cache_big/w{w}-mlp-256x4.npz")[0]
    for k, A in (("va", SB_va), ("te", SB_te)):
        Y = np.asarray(ml.mlp_forward(params, xw.transform(A[:, w, :])))
        gh[k][:, :, w], cf[k][:, :, w] = ml.phasor_decode(Y, lw)
    g_tr = cw.inverse(np.asarray(ml.mlp_forward(params, xw.transform(SB_tr[:, w, :]))))
    sig_tr[w] = np.sqrt(np.mean(ml.wrap(g_tr - ml.wrap(CB_tr, lw), lw) ** 2, axis=0))
res = []
for k, C_e in (("va", CB_va), ("te", CB_te)):
    fit = dict(ghat_va=gh[k], conf_va=cf[k], sig_tr=sig_tr, periods=lam, wl_use=np.arange(len(lam)))
    P, _ = ml.unwrap_all(fit, SPAN)
    res.append((ml.rmse_nm(P, C_e).mean(), 100 * np.mean(np.abs(P - C_e) > lam.min() / 2)))
spectral["MLP 4x256"] = res

print("\nspectral, +-8 lambda, 120 nm SCAO residual, each method + unwrap: RMSE (nm) [wrong fringes %]")
print(f"{'method':24s}{'validation':>22s}{'test':>22s}")
for m, ((v, bv), (t, bt)) in spectral.items():
    print(f"{m:24s}{v:14.2f} [{bv:5.2f}%]{t:14.2f} [{bt:5.2f}%]")
np.savez(f"{DATA}/test_scores.npz", single=np.array([single], dtype=object),
         spectral=np.array([spectral], dtype=object))
print("saved", f"{DATA}/test_scores.npz")
