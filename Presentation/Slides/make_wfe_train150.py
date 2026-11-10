#!/usr/bin/env python
"""Wavefront-error figure with ONE training level (19 ports, 1200 nm, single wavelength).

Every reconstructor is trained once, on the 150 nm SCAO-residual dataset (single_ao150.npz, its
8000 training rows; the OMP budget K chosen on its 2000 validation rows), and then tested, without
retraining, on the independent test set of every level (single_ao<r>_test.npz, r = 0 ... 150 nm:
2000 new piston configurations, each with its own residual screen). The fits are the ones cached
by trainMLP.ipynb (cache_single_phasor, tag ao150-*) and by the 4x256 MLP run (cache_big); the
symbolic expressions found at 150 nm (scripts/symbolic_sr.py, data/ml/pl19_1200/symbolic_ao150)
are re-evaluated on the test signals (checked against their stored validation predictions).

    python make_wfe_train150.py   -> figures/slides/pl19_1200_wfe_train150.pdf
                                     ../../data/ml/pl19_1200/wfe_train150_scores.npz
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import slidefig                                                         # noqa: E402
import ml_reconstruct as ml                                             # noqa: E402
from ml_reconstruct import forward as fw, petals, symbolic as sy        # noqa: E402
from ml_reconstruct.presets import settings                             # noqa: E402
from ml_reconstruct.simulate import bp                                  # noqa: E402

os.environ.setdefault("CBEAM_LANTERN", "pl19_1200")
S = settings()
DATA = os.path.join(ROOT, "data", "ml", "pl19_1200")
LAM = 1200.0
TRAIN = 150
LEVELS = [0, 30, 60, 90, 120, 150]
KS = S["wfe_ks"]
ys = ml.PhasorCodec(LAM)
cache = ml.ModelCache(os.path.join(DATA, "cache_single_phasor"), verbose=False)
_p = bp.get_simulation_parameters(lantern="pl19_1200")
_f = bp.load_ifunc(_p["ifunc_file"]); bp.center_ifunc_mask(_f)
_g = petals.petal_geometry(np.asarray(_f.mask_inf_func) > 0, np.asarray(_f.influence_function), 5)
TWIN = petals.twin_matrix(_g["opposite"], _g["ref"])

tr = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{TRAIN}.npz"))
S_tr, C_tr = tr.train
S_va, C_va = tr.val
xs = ml.Standardizer().fit(S_tr)
X_tr = xs.transform(S_tr)
tag = f"ao{TRAIN}"
models = {"linear": cache(f"{tag}-linear", None), "d2": cache(f"{tag}-d2", None),
          "d3": cache(f"{tag}-d3", None), "cf": cache(f"{tag}-closed-form", None)}
omp = {K: cache(f"{tag}-omp-K{K}", None) for K in KS}
mlp = ml.load_mlp(os.path.join(DATA, "cache_big", f"{tag}-mlp-256x4.npz"))[0]

# ----------------------------------------------------------------- symbolic regression at 150 nm
PETAL_ANGLES_DEG = (120.0, 180.0, -119.8, -59.8, 0.0, 59.6)       # scripts/symbolic_sr.py
REF = 5
SR_FIT, SR_VA = slice(0, 6000), slice(8000, 10000)                  # its SR_SPLITS 0,6000,8000,10000
SR_DIR = os.path.join(DATA, f"symbolic_ao{TRAIN}")
NS = {n: getattr(np, n) for n in ("sin", "cos", "exp", "log", "sqrt", "abs", "tanh", "arctan", "square")}
NS.update(aq=lambda a, b: a / np.sqrt(1 + b ** 2), pow=np.power)


def pair_frame(k):
    a = np.radians(PETAL_ANGLES_DEG)
    d = np.angle(np.exp(1j * (a[k] - a[REF])))
    return a[REF] + d / 2, (np.sign(d) if d != 0 else 1.0)


def _eval(expr, F, names):
    env = dict(NS); env.update({n: F[:, i] for i, n in enumerate(names)})
    return np.broadcast_to(eval(expr.replace("^", "**"), {"__builtins__": {}}, env), (len(F),))


def symbolic_reconstructor():
    """predict(S) -> pistons [nm] with the expressions sr_reconstructor picks; None if absent."""
    best = sy.sr_reconstructor(SR_DIR, LAM)
    if best is None:
        return None
    engine, P_va_stored, _, rows = best
    res = sy.load_sr(SR_DIR, engine)
    ds = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{TRAIN}.npz"))
    xy = ds.core_xy
    stats = {}
    for c in rows:                                   # the standardisation of each class: its fit rows
        F = np.concatenate([sy.harmonic_features(ds.signals[SR_FIT], xy, *pair_frame(k))[0]
                            for k in sy.CLASSES[c]])
        stats[c] = (F.mean(0), F.std(0))
    names = list(res[(next(iter(rows)), "sin")]["names"])

    def predict(Sig):
        P = np.zeros((len(Sig), 5))
        for k, (c, _) in sy.PISTON_CLASS.items():
            mu, sd = stats[c]
            F = (sy.harmonic_features(Sig, xy, *pair_frame(k))[0] - mu) / sd
            r = rows[c]
            s_hat = _eval(str(res[(c, "sin")]["expression"][int(r.i)]), F, names)
            c_hat = _eval(str(res[(c, "cos")]["expression"][int(r.j)]), F, names)
            P[:, k] = sy.decode(s_hat, c_hat, LAM)
        return P

    chk = predict(ds.signals[SR_VA])
    dev = np.abs(ml.wrap(chk - P_va_stored, LAM)).max()
    print(f"symbolic ({engine}): re-evaluated vs stored validation predictions, max |diff| {dev:.3g} nm")
    if dev > 1.0:
        raise SystemExit("the re-evaluated symbolic expressions do not reproduce the stored predictions")
    return predict


SYM = symbolic_reconstructor()


def predict(Sig):
    X = xs.transform(Sig)
    F = ml.poly_features(X_tr, X, 3)[1]
    out = {"linear": ys.inverse(models["linear"].predict(X)),
           "dense d2": ys.inverse(ml.predict_dense_poly_ridge(models["d2"], X)),
           "dense d3": ys.inverse(ml.predict_dense_poly_ridge(models["d3"], X)),
           "closed form": fw.invert_fixed(models["cf"], Sig, TWIN)[0],
           "mlp": ys.inverse(np.asarray(ml.mlp_forward(mlp, X)))}
    for K in KS:
        out[f"omp{K}"] = ys.inverse(np.stack([ml.omp_predict(F, f) for f in omp[K]], axis=1))
    if SYM is not None:
        out["symbolic"] = SYM(Sig)
    return out


rmse = lambda P, C: float(ml.rmse_nm(P, C[:, :5], LAM).mean())
val = {m: rmse(P, C_va) for m, P in predict(S_va).items()}
KBEST = min(KS, key=lambda K: val[f"omp{K}"])
print(f"trained at {TRAIN} nm, validation:", {m: round(v, 2) for m, v in val.items()}, " OMP K =", KBEST)
scores = {}
for r in LEVELS:
    te = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{r}_test.npz"))
    for m, P in predict(te.signals).items():
        scores.setdefault(m, {})[r] = rmse(P, te.coeff)
    print(f"test {r:3d} nm: " + "  ".join(f"{m} {scores[m][r]:.2f}" for m in scores), flush=True)
scores["omp"] = scores[f"omp{KBEST}"]
np.savez(os.path.join(DATA, "wfe_train150_scores.npz"), levels=LEVELS, kbest=KBEST,
         scores=np.array([scores], dtype=object), validation=np.array([val], dtype=object))

# ------------------------------------------------------------------ the figure (slot 7.6 x 5.6 cm)
P = slidefig.PALETTE
plt.rcParams.update(slidefig.RC)
SHOW = [("linear", "linear", "s-"), ("symbolic", "symbolic regression", "v-"), ("dense d2", "dense poly, degree 2", "^-"),
        ("dense d3", "dense poly, degree 3", "<-"), ("omp", f"sparse poly (OMP), K={KBEST}", "o-"),
        ("closed form", "closed form", "P-"), ("mlp", "MLP", "D-")]
fig, ax = plt.subplots(figsize=(7.6 * slidefig.CM, 5.6 * slidefig.CM), layout="constrained")
for key, label, style in SHOW:
    if key in scores:
        ax.semilogy(LEVELS, [scores[key][r] for r in LEVELS], style, color=P[key], label=label, ms=3, lw=1)
ax.axvline(TRAIN, color="0.5", lw=0.8, ls=":")
ax.text(TRAIN - 2, 0.13, "training\nlevel", ha="right", va="bottom", color="0.4")
ax.set_xticks(LEVELS); ax.set_ylim(bottom=0.1)
ax.set_xlabel("wavefront error besides the pistons (nm rms, SCAO residual)")
ax.set_ylabel("RMSE (nm)")
ax.grid(alpha=0.3, which="both")
fig.legend(loc="outside lower center", ncol=2, frameon=False)
slidefig.export(fig, "pl19_1200_wfe_train150", native=True)
