"""The multi-wavelength unwrap of PL_EML.ipynb on a smaller piston range, from
the cached per-wavelength models: the validation samples whose pistons all lie
within +-span_nm, unwrapped with the search range set to +-span_nm.

The per-wavelength models regress the phasor of each piston, so they do not
depend on the range; only the unwrap does. This gives the result at a smaller
physical range without a new dataset (fewer samples: those inside the range).

    python scripts/unwrap_subrange.py <lantern> <span_nm>    # e.g. pl19_1200 6400
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
LANTERN, SPAN = sys.argv[1], float(sys.argv[2])
os.environ["CBEAM_LANTERN"] = LANTERN
import ml_reconstruct as ml                       # noqa: E402
from ml_reconstruct.presets import settings      # noqa: E402

S = settings(LANTERN)
DATA = S["data_dir"]
ds = ml.LanternDataset.load(S["spectral"])
S_tr, C_tr = ds.train
S_va, C_va = ds.val
lam = ds.wavelengths_nm
xs = ml.Standardizer().fit(S_tr.reshape(len(S_tr), -1))
X3_tr = xs.transform(S_tr.reshape(len(S_tr), -1)).reshape(S_tr.shape)
X3_va = xs.transform(S_va.reshape(len(S_va), -1)).reshape(S_va.shape)
cache = ml.ModelCache(f"{DATA}/cache_perlambda", retrain=False)
WL_USE = np.arange(0, len(lam), 2)
K = S["k_wrap"]
rows = ml.subsample_rows(len(X3_tr), 0.2, seed=0)
fit = ml.fit_wrapped_models(X3_tr, X3_va, C_tr, C_va, lam, WL_USE, np.arange(S_tr.shape[2]),
                            degree=3, K=K, screen=2000, cache=cache, rows=rows,
                            tag=f"wrap1-d3-K{K}-f{S_tr.shape[2]}-n{len(rows)}")


def score(sel, span):
    sub = dict(fit, ghat_va=fit["ghat_va"][sel], conf_va=fit["conf_va"][sel])
    P, _ = ml.unwrap_all(sub, span)
    e = P - C_va[sel]
    wrong = np.abs(e) > fit["periods"].min() / 2
    rmse = np.sqrt(np.mean(e ** 2, axis=0))
    bulk = np.sqrt(np.mean(np.where(wrong, 0, e) ** 2, axis=0) / np.maximum(1e-12, 1 - wrong.mean(0)))
    return rmse, wrong, bulk


full = S["spectral_span_nm"]
for name, sel, span in (("full range", np.ones(len(C_va), bool), full),
                        (f"+-{SPAN:g} nm", np.abs(C_va).max(1) <= SPAN, SPAN)):
    rmse, wrong, bulk = score(sel, span)
    print(f"{LANTERN} {name:14s} ({sel.sum()} samples, +-{span / lam.mean():.2f} lambda): "
          f"RMSE {rmse.mean():6.2f} nm {np.round(rmse, 2)}, wrong fringes {int(wrong.sum())} "
          f"({100 * wrong.mean():.3f}%), without them {bulk.mean():.2f} nm")
