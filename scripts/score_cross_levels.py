"""Offline cross-test of the per-wavelength models: trained at 120 or 213 nm, tested on both test sets."""
import os, sys, numpy as np
ROOT = "/raid1/frossi/dev/work/PLREC"; sys.path.insert(0, ROOT); os.chdir(ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import ml_reconstruct as ml
from ml_reconstruct import petals
from ml_reconstruct.presets import settings
from ml_reconstruct.simulate import bp
S = settings(); DATA = S["data_dir"]
_p = bp.get_simulation_parameters(lantern="pl19_1200"); _f = bp.load_ifunc(_p["ifunc_file"]); bp.center_ifunc_mask(_f)
_g = petals.petal_geometry(np.asarray(_f.mask_inf_func) > 0, np.asarray(_f.influence_function), 5)
TWIN = petals.twin_matrix(_g["opposite"], _g["ref"])
tests = {r: ml.LanternDataset.load(f"{DATA}/spectral_ao{r}_21wl_test.npz") for r in (120, 213)}
for rtr in (120, 213):
    tr = ml.LanternDataset.load(f"{DATA}/spectral_ao{rtr}_21wl.npz"); lam = np.asarray(tr.wavelengths_nm, float)
    SB_tr, CB_tr = tr.train
    xs3 = ml.Standardizer().fit(SB_tr.reshape(len(SB_tr), -1)); X3 = lambda A: xs3.transform(A.reshape(len(A), -1)).reshape(A.shape)
    cache = ml.ModelCache(f"{DATA}/cache_ao{rtr}_perlambda", verbose=False)
    for kind, tag in (("closed_form", "closed_form"), ("omp", "omp-K1282")):
        for rte, te in tests.items():
            Ce = te.coeff[:, :5]
            ft = ml.fit_wrapped_models(X3(SB_tr), X3(te.signals), CB_tr, Ce, lam, np.arange(len(lam)), np.arange(19), degree=3,
                                       K=1282, screen=2000, cache=cache, tag=tag, model=kind, twin=TWIN)
            P, _ = ml.unwrap_all(ft, S["spectral_span_nm"])
            print(f"trained {rtr} {kind:12s} tested {rte}: per-lambda wrapped {ft['sig_va'].mean():6.1f} nm, "
                  f"unwrapped {ml.rmse_nm(P, Ce).mean():7.1f} nm, wrong fringes {100*np.mean(np.abs(P-Ce) > lam.min()/2):5.2f} %", flush=True)
