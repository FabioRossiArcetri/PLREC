"""Symmetry-adapted features of the lantern output for compact (symbolic)
reconstructors at one wavelength (symbolic.ipynb).

Physics: with piston-only aberrations and monochromatic light, the field at
each output core is linear in the six petal phasors exp(i phi_j), so every
output power is a Hermitian quadratic form in them -- linear in the 30
features cos / sin of the 15 pairwise phase differences. A reconstructor has
to invert that trigonometric map; regressing each piston's phasor
(sin, cos)(2 pi p / lambda) and taking atan2 avoids having to build the
arctangent into the expression.

Features: the 19 output cores lie on rings of a hexagonal grid (centre; 6 at
distance a; 6 at sqrt(3) a; 6 at 2 a). The angular harmonics of each ring,
sum_c s_c exp(-i m theta_c), m = 0..3, re-express the 19 signals as 19 numbers
ordered by symmetry. For piston k they are taken in the frame of the bisector
between petal k and the reference petal, oriented so that petal k is at
+half its angle from the reference: their real parts are even, their
imaginary parts odd, under the mirror that swaps petal k and the reference.
"""
from __future__ import annotations

import numpy as np


def rings(core_xy, tol=0.05):
    """(ring index per core, angle per core [rad]): ring 0 = centre, then the
    rings in order of distance from the centre."""
    xy = np.asarray(core_xy, float)
    r = np.hypot(xy[:, 0], xy[:, 1])
    levels = []
    for v in np.sort(r):
        if not levels or v > levels[-1] * (1 + tol) + 1e-9:
            levels.append(v)
    ring = np.array([int(np.argmin(np.abs(np.array(levels) - v))) for v in r])
    return ring, np.arctan2(xy[:, 1], xy[:, 0])


def petal_angles_mesh(geom):
    """Petal centre angles [rad] in the lantern's (x, y) frame: the pupil's image
    rows map onto +y of the focal plane and of the lantern mesh, so the angle is
    the negative of petal_geometry's image angle (counted with rows down)."""
    return -np.radians(geom["angle_deg"])


def pair_frame(geom, k):
    """(frame angle, orientation sign) for piston k: the bisector of petal k and
    the reference, oriented so that petal k is on the positive side."""
    a = petal_angles_mesh(geom)
    ak, ar = a[k], a[geom["ref"]]
    d = np.angle(np.exp(1j * (ak - ar)))                 # signed angle from the reference to petal k
    return ar + d / 2, np.sign(d) if d != 0 else 1.0


def harmonic_features(S, core_xy, frame_angle=0.0, orientation=1.0, normalize=True, m_max=3):
    """(n, 19) angular-harmonic features of the signals S (n, 19) in a frame
    rotated by ``frame_angle`` (and mirrored if orientation < 0), and their names.

    Per ring of 6 cores: m = 0 (real), m = 1, 2 (real, imaginary), m = 3 (the
    component along the frame: its real part); plus the centre core. normalize:
    divide by the total signal first (removes the overall coupled power).
    """
    S = np.asarray(S, float)
    if normalize:
        S = S / S.sum(axis=1, keepdims=True)
    ring, theta = rings(core_xy)
    th = orientation * (theta - frame_angle)              # angles in the (possibly mirrored) frame
    feats, names = [S[:, ring == 0].sum(1)], ["c0"]
    for rg in range(1, ring.max() + 1):
        sel = ring == rg
        for m in range(m_max + 1):
            h = S[:, sel] @ np.exp(-1j * m * th[sel])
            if m == 0:
                feats.append(h.real); names.append(f"r{rg}m0")
            elif m < m_max or sel.sum() > 2 * m:
                feats += [h.real, h.imag]; names += [f"r{rg}m{m}re", f"r{rg}m{m}im"]
            else:                                         # m = 3 on 6 points: one real dof
                feats.append(h.real); names.append(f"r{rg}m{m}re")
    F = np.stack(feats, axis=1)
    return F, names


def phasor_targets_k(P, k, wavelength_nm):
    """(n, 2) sin and cos of 2 pi p_k / lambda."""
    ph = 2 * np.pi * np.asarray(P)[:, k] / wavelength_nm
    return np.stack([np.sin(ph), np.cos(ph)], axis=1)


def decode(sin_hat, cos_hat, wavelength_nm):
    """Piston [nm] from estimated sin and cos of its phase."""
    return np.arctan2(sin_hat, cos_hat) * wavelength_nm / (2 * np.pi)


# --------------------------------------------------------------------------- #
# symbolic-regression results (scripts/symbolic_sr.py)
# --------------------------------------------------------------------------- #
CLASSES = {"opposite": [2], "adjacent": [0, 4], "next": [1, 3], "adjacent-0": [0], "adjacent-4": [4]}
# the classes that make a full 5-piston reconstructor: piston -> (class, member index)
PISTON_CLASS = {0: ("adjacent-0", 0), 1: ("next", 0), 2: ("opposite", 0), 3: ("next", 1), 4: ("adjacent-4", 0)}


def _wrapped(e, wavelength_nm):
    return e - wavelength_nm * np.round(e / wavelength_nm)


def count_constants(expr, names):
    """Number of numeric constants in an expression string (the digits inside
    the variable names, e.g. r1m2re, are not counted)."""
    import re
    for n in sorted(names, key=len, reverse=True):
        expr = expr.replace(n, "V")
    return len(re.findall(r"(?<![A-Za-z_])\d+\.?\d*(?:[eE][-+]?\d+)?", expr))


def load_sr(directory, engine):
    """{(class, 'sin' | 'cos'): npz contents} of one engine's results."""
    import os
    out = {}
    for c in CLASSES:
        for t in ("sin", "cos"):
            path = os.path.join(directory, f"{engine}_{c}_{t}.npz")
            if os.path.exists(path):
                with np.load(path) as f:
                    out[(c, t)] = {k: f[k] for k in f.files}
    return out


def sr_front(res, cls, wavelength_nm):
    """Accuracy vs. complexity of the (sin, cos) expression pairs of a class:
    for every pair, the piston RMSE [nm] (decoded with atan2) on the held-out
    and validation rows; then the Pareto front on (total complexity, held-out
    RMSE). Returns a DataFrame sorted by complexity, with the expressions and
    their constant counts."""
    import pandas as pd
    s, c = res[(cls, "sin")], res[(cls, "cos")]
    true_es = decode(s["true_es"], c["true_es"], wavelength_nm)
    true_va = decode(s["true_va"], c["true_va"], wavelength_nm)
    names = list(s["names"])
    rows = []
    for i in range(len(s["complexity"])):
        for j in range(len(c["complexity"])):
            # errors modulo lambda: an estimate across the wrap is the right piston
            e_es = _wrapped(decode(s["pred_es"][i], c["pred_es"][j], wavelength_nm) - true_es, wavelength_nm)
            rows.append(dict(complexity=int(s["complexity"][i] + c["complexity"][j]), i=i, j=j,
                             es_rmse=np.sqrt(np.mean(e_es ** 2))))
    df = pd.DataFrame(rows).sort_values(["complexity", "es_rmse"])
    front, best = [], np.inf
    for _, r in df.iterrows():
        if r.es_rmse < best:
            best = r.es_rmse
            i, j = int(r.i), int(r.j)
            e_va = _wrapped(decode(s["pred_va"][i], c["pred_va"][j], wavelength_nm) - true_va, wavelength_nm)
            front.append(dict(complexity=int(r.complexity), i=i, j=j, es_rmse=r.es_rmse,
                              val_rmse=np.sqrt(np.mean(e_va ** 2)),
                              val_median=np.median(np.abs(e_va)),
                              constants=count_constants(str(s["expression"][i]), names)
                              + count_constants(str(c["expression"][j]), names),
                              sin_expr=str(s["expression"][i]), cos_expr=str(c["expression"][j])))
    return pd.DataFrame(front)


def sr_reconstructor(directory, wavelength_nm, engines=("operon", "pysr")):
    """The best symbolic 5-piston reconstructor found: for each engine, per
    class of PISTON_CLASS the (sin, cos) expression pair with the lowest
    held-out RMSE (modulo lambda) on its front; the engine with the lower mean
    held-out RMSE wins. Returns (engine, P_va (n_va, 5) [nm, in -lambda/2..],
    constants per piston (5,), {class: front row}) or None without results."""
    best = None
    for engine in engines:
        res = load_sr(directory, engine)
        classes = {c for c, _ in PISTON_CLASS.values()}
        if not all((c, t) in res for c in classes for t in ("sin", "cos")):
            continue
        rows = {}
        for c in classes:
            f = sr_front(res, c, wavelength_nm)
            rows[c] = f.loc[f.es_rmse.idxmin()]
        n_va = len(res[("opposite", "sin")]["true_va"])
        P = np.zeros((n_va, 5)); const = np.zeros(5)
        for k, (c, m) in PISTON_CLASS.items():
            r = rows[c]
            sl = slice(m * n_va, (m + 1) * n_va)
            P[:, k] = decode(res[(c, "sin")]["pred_va"][int(r.i)][sl], res[(c, "cos")]["pred_va"][int(r.j)][sl],
                             wavelength_nm)
            const[k] = r.constants
        score = np.mean([rows[c].es_rmse for c in rows])
        if best is None or score < best[0]:
            best = (score, engine, P, const, rows)
    return None if best is None else best[1:]


def plot_sr_fronts(fronts, refs=None):
    """Validation RMSE vs. total complexity of each {label: sr_front} (log y),
    with horizontal reference lines {label: rmse}."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(9, 4.8))
    for i, (label, f) in enumerate(fronts.items()):
        if len(f):
            plt.plot(f.complexity, f.val_rmse, "o-", ms=3, label=label, color=plt.cm.tab10(i))
    for j, (label, v) in enumerate((refs or {}).items()):
        plt.axhline(v, ls="--", lw=1, color=plt.cm.Dark2(j), label=label)
    plt.yscale("log"); plt.xlabel("total complexity (nodes of the sin + cos expressions)")
    plt.ylabel("validation RMSE of the piston (nm)"); plt.grid(alpha=0.3, which="both")
    plt.legend(fontsize=7, ncol=2); plt.title("symbolic reconstructors: accuracy vs. complexity")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# explicit polynomial baselines of the phasors
# --------------------------------------------------------------------------- #
def fit_phasor_poly(F, T, degree, K=None):
    """Polynomial of degree ``degree`` in the features F for the (sin, cos)
    targets T: ridge on every term (K None) or OMP with at most K terms per
    target. Returns (poly, scaler, [model_sin, model_cos])."""
    from sklearn.linear_model import OrthogonalMatchingPursuit, RidgeCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    pf = PolynomialFeatures(degree, include_bias=False)
    Z = pf.fit_transform(F); sc = StandardScaler().fit(Z); Z = sc.transform(Z)
    make = (lambda: RidgeCV(alphas=np.logspace(-6, 2, 9))) if K is None else \
        (lambda: OrthogonalMatchingPursuit(n_nonzero_coefs=K))
    return pf, sc, [make().fit(Z, T[:, j]) for j in range(2)]


def predict_phasor_poly(model, F):
    pf, sc, ms = model
    Z = sc.transform(pf.transform(F))
    return np.stack([m.predict(Z) for m in ms], axis=1)


def phasor_poly_dof(model):
    """Fitted constants per piston: non-zero coefficients and intercepts of both
    (sin, cos) targets of one piston."""
    return int(sum(np.count_nonzero(m.coef_) + 1 for m in model[2]))


def load_sr_direct(directory, engine, span_nm):
    """{class: npz contents} of one engine's direct-piston results on the
    +-span_nm synthetic problem (scripts/symbolic_sr.py --range)."""
    import os
    out = {}
    for c in CLASSES:
        path = os.path.join(directory, f"{engine}_{c}_r{span_nm:g}_p.npz")
        if os.path.exists(path):
            with np.load(path) as f:
                out[c] = {k: f[k] for k in f.files}
    return out


def sr_front_direct(r):
    """Pareto front (complexity, held-out RMSE) of direct piston expressions,
    with the validation RMSE and the constant counts."""
    import pandas as pd
    names = list(r["names"])
    es = np.sqrt(np.mean((r["pred_es"] - r["true_es"]) ** 2, axis=1))
    va = np.sqrt(np.mean((r["pred_va"] - r["true_va"]) ** 2, axis=1))
    rows, best = [], np.inf
    for i in np.argsort(r["complexity"], kind="stable"):
        if es[i] < best:
            best = es[i]
            rows.append(dict(complexity=int(r["complexity"][i]), es_rmse=es[i], val_rmse=va[i],
                             constants=count_constants(str(r["expression"][i]), names),
                             expression=str(r["expression"][i])))
    return pd.DataFrame(rows)


def range_scan(model, spans_nm, Ks=(10, 30), n_fit=20000, n_va=5000, rng=3):
    """Explicit reconstructors vs. capture range, on noise-free signals of the
    exact forward model: linear (ridge, all 19 signals) and sparse quadratic
    (OMP, K terms per piston), each fitted to the piston directly. Returns the
    validation RMSE [nm] (mean over the pistons) and the degrees of freedom
    per piston."""
    import pandas as pd
    from sklearn.linear_model import OrthogonalMatchingPursuit, RidgeCV
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler
    g = np.random.default_rng(rng)
    rows = []
    for R in spans_nm:
        Ptr, Pva = g.uniform(-R, R, (n_fit, 5)), g.uniform(-R, R, (n_va, 5))
        xs = StandardScaler().fit(model(Ptr)); Xtr, Xva = xs.transform(model(Ptr)), xs.transform(model(Pva))
        lin_model = RidgeCV(alphas=np.logspace(-10, 0, 11)).fit(Xtr, Ptr)
        lin = lin_model.predict(Xva)
        n_lin = int(np.shape(lin_model.coef_)[-1] + 1)        # per piston
        row = {"range_nm": R, f"linear ({n_lin} dof/piston)": np.sqrt(np.mean((lin - Pva) ** 2))}
        pf = PolynomialFeatures(2, include_bias=False); sq = StandardScaler().fit(pf.fit_transform(Xtr))
        Qtr, Qva = sq.transform(pf.transform(Xtr)), sq.transform(pf.transform(Xva))
        for K in Ks:
            pr = np.stack([OrthogonalMatchingPursuit(n_nonzero_coefs=K).fit(Qtr, Ptr[:, k]).predict(Qva)
                           for k in range(5)], 1)
            row[f"quadratic K={K} ({K + 1} dof/piston)"] = np.sqrt(np.mean((pr - Pva) ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# the best cubic model of workbench.ipynb Part A
# --------------------------------------------------------------------------- #
def workbench_zern_omp(S_train, C_train, core_xy, S_eval, cache_dir="data/ml/cache_single", K=1282,
                       n_zern=36):
    """Estimates [nm] on S_eval of workbench.ipynb's best cubic model: OMP with
    K terms per piston on the degree-3 monomials of the standardised Zernike
    projection of the core signals, loaded from its cache. S_train, C_train:
    the rows it was trained on (the 8000 training rows of the 10k dataset), to
    rebuild its scalers exactly. Returns (estimates, constants per piston)."""
    from . import ModelCache, Standardizer, modal_projector, poly_features, zernike_basis, omp_predict
    proj = modal_projector(zernike_basis(core_xy, n_zern, fill=0.8)[0])
    zs = Standardizer(eps=1e-9).fit(S_train @ proj.T)
    Z_tr, Z_ev = zs.transform(S_train @ proj.T), zs.transform(S_eval @ proj.T)
    _, Zd3_ev, _, _ = poly_features(Z_tr, Z_ev, 3)
    fits = ModelCache(cache_dir, verbose=False)(f"zern-omp-K{K}", None)
    ys = Standardizer().fit(C_train)
    pred = ys.inverse(np.stack([omp_predict(Zd3_ev, f) for f in fits], axis=1))
    constants = int(round(np.mean([np.count_nonzero(f[0].coef_) + 1 for f in fits])))
    return pred, constants          # constants PER PISTON


def workbench_fiber_omp(S_train, C_train, S_eval, cache_dir="data/ml/cache_single", K=1282):
    """Estimates [nm] on S_eval of workbench.ipynb's best cubic model: OMP with
    K terms per piston on the degree-3 monomials of the standardised core
    signals, loaded from its cache. S_train, C_train: the rows it was trained on
    (the 8000 training rows of the 10k dataset), to rebuild its scalers exactly.
    Returns (estimates, constants per piston).

    The Zernike variant kept above (``workbench_zern_omp``) is retired from the
    notebooks; its cached fits are still on disk.

    K = 1282 is the saturated OMP budget on this basis (1281 non-zero monomials
    plus the intercept per piston): every larger budget, up to the full cubic
    basis, gives the identical fit."""
    from . import ModelCache, Standardizer, poly_features, omp_predict
    xs = Standardizer().fit(S_train)
    X_tr, X_ev = xs.transform(S_train), xs.transform(S_eval)
    _, X3_ev, _, _ = poly_features(X_tr, X_ev, 3)
    fits = ModelCache(cache_dir, verbose=False)(f"fiber-omp-K{K}", None)
    ys = Standardizer().fit(C_train)
    pred = ys.inverse(np.stack([omp_predict(X3_ev, f) for f in fits], axis=1))
    constants = int(round(np.mean([np.count_nonzero(f[0].coef_) + 1 for f in fits])))
    return pred, constants          # constants PER PISTON


def sr_front_template(r, wavelength_nm):
    """Pareto front (complexity, held-out piston RMSE) of template expressions
    theta = atan2(f, g) (scripts/symbolic_template.py): errors wrapped to +-pi,
    in nm."""
    import pandas as pd
    nm = wavelength_nm / (2 * np.pi)
    err = lambda p, t: np.sqrt(np.mean((np.angle(np.exp(1j * (p - t))) * nm) ** 2, axis=-1))
    es, va = err(r["pred_es"], r["true_es"]), err(r["pred_va"], r["true_va"])
    names = list(r["names"])
    rows, best = [], np.inf
    for i in np.argsort(r["complexity"], kind="stable"):
        if es[i] < best:
            best = es[i]
            rows.append(dict(complexity=int(r["complexity"][i]), es_rmse=es[i], val_rmse=va[i],
                             constants=count_constants(str(r["expression"][i]), names),
                             expression=str(r["expression"][i])))
    return pd.DataFrame(rows)
