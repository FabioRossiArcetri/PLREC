"""Symbolic regression of single-wavelength reconstructors (symbolic.ipynb).

For each class of piston (by its petal's position relative to the reference:
"opposite" = piston 2; "adjacent" = pistons 0 and 4; "next" = pistons 1 and 3,
pooled through the mirror symmetry), fits explicit expressions of the 19
angular-harmonic features of the 800 nm camera signals
(ml_reconstruct.symbolic.harmonic_features, in each piston's pair frame) for
the sine and the cosine of the piston's phase 2 pi p / lambda; the piston is
then atan2(sin, cos) lambda / 2 pi.

Engines: Operon (pyoperon, multi-objective: accuracy vs. length, the whole
Pareto front) and PySR (its hall of fame over complexities). Every expression
of each front is saved with its predictions on the held-out (es) and
validation rows, so the notebook can pair sine and cosine models of a given
complexity and score the decoded pistons.

--range R instead fits the piston itself over a capture range of +-R nm, on
noise-free data synthesised from the exact forward model
(ml_reconstruct.forward, fitted to the dataset): uniform pistons in +-R,
N_SR fit rows per class member, 4000 held-out and 10000 validation rows.

    python scripts/symbolic_sr.py [--engine {operon,pysr,all}] [--range R]
      -> data/ml/symbolic/<engine>_<class>_<sin|cos>.npz, or
         data/ml/symbolic/<engine>_<class>_r<R>_p.npz  (restartable)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml                       # noqa: E402
from ml_reconstruct import symbolic as sy         # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "symbolic")
DATASET = os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")
LAMBDA_NM = 800.0
FIT, ES, VA = slice(0, 36000), slice(36000, 40000), slice(40000, 50000)
N_SR = 10000                      # fit rows per class member given to the engines
CLASSES = {"opposite": [2], "adjacent": [0, 4], "next": [1, 3],
           "adjacent-0": [0]}            # piston 0 alone: pooling 0 and 4 through the mirror fails
PYSR_SECONDS = 600
OPERON_SECONDS = 600
N_THREADS = 32
# Another lantern (e.g. pl19_1200 on its 10000-sample single-wavelength dataset):
#   SR_DATASET=data/ml/pl19_1200/single.npz SR_LAMBDA=1200 SR_SPLITS=0,6000,8000,10000
#   SR_OUT=data/ml/pl19_1200/symbolic SR_THREADS=16 python scripts/symbolic_sr.py ...
# (adds the class "adjacent-4", so that every piston has a class of its own or a pooled one)
if os.environ.get("SR_DATASET"):
    DATASET = os.path.join(ROOT, os.environ["SR_DATASET"])
    LAMBDA_NM = float(os.environ["SR_LAMBDA"])
    a_, b_, c_, d_ = (int(v) for v in os.environ["SR_SPLITS"].split(","))
    FIT, ES, VA = slice(a_, b_), slice(b_, c_), slice(c_, d_)
    N_SR = b_ - a_
    OUT = os.path.join(ROOT, os.environ["SR_OUT"])
    N_THREADS = int(os.environ.get("SR_THREADS", N_THREADS))
    CLASSES = {"opposite": [2], "next": [1, 3], "adjacent-0": [0], "adjacent-4": [4]}
# Petal angles in the lantern frame (ml_reconstruct.symbolic.petal_angles_mesh of
# the bundled pupil), so that this script does not need cbeam.
PETAL_ANGLES_DEG = (120.0, 180.0, -119.8, -59.8, 0.0, 59.6)
REF = 5


def pair_frame(k):
    a = np.radians(PETAL_ANGLES_DEG)
    d = np.angle(np.exp(1j * (a[k] - a[REF])))
    return a[REF] + d / 2, (np.sign(d) if d != 0 else 1.0)


def synthetic_dataset(span_nm, n=N_SR + 14000, seed=7):
    """(signals, pistons, core_xy) uniform in +-span_nm, from the exact forward model."""
    from ml_reconstruct import forward as fw
    ds = ml.LanternDataset.load(DATASET)
    model = fw.fit_forward(ds.coeff[:10000], ds.signals[:10000], LAMBDA_NM)
    P = np.random.default_rng(seed).uniform(-span_nm, span_nm, (n, 5))
    return model(P), P, ds.core_xy


def class_data(name, span_nm=None):
    """Standardised features and targets of a class, per split; the fit rows of
    the members are stacked (N_SR each). Targets: (sin, cos) of the phase on the
    dataset (span_nm None), or the piston [nm] on synthetic data in +-span_nm."""
    if span_nm is None:
        ds = ml.LanternDataset.load(DATASET)
        S_all, P_all, xy, splits = ds.signals, ds.coeff, ds.core_xy, (("fit", FIT), ("es", ES), ("va", VA))
    else:
        S_all, P_all, xy = synthetic_dataset(span_nm)
        splits = (("fit", slice(0, N_SR)), ("es", slice(N_SR, N_SR + 4000)), ("va", slice(N_SR + 4000, None)))
    out = {}
    for split, rows in splits:
        F, T = [], []
        for k in CLASSES[name]:
            a, o = pair_frame(k)
            f, names = sy.harmonic_features(S_all[rows], xy, a, o)
            t = (sy.phasor_targets_k(P_all[rows], k, LAMBDA_NM) if span_nm is None
                 else P_all[rows][:, [k]])
            if split == "fit":
                f, t = f[:N_SR], t[:N_SR]
            F.append(f); T.append(t)
        out[split] = (np.concatenate(F), np.concatenate(T))
    mu, sd = out["fit"][0].mean(0), out["fit"][0].std(0)
    out = {s: ((F - mu) / sd, T) for s, (F, T) in out.items()}
    return out, names, (mu, sd)


def run_operon(X, y, names):
    from pyoperon.sklearn import SymbolicRegressor
    reg = SymbolicRegressor(allowed_symbols="add,sub,mul,div,square,constant,variable", max_length=60,
                            generations=2000, population_size=2000, pool_size=2000, max_evaluations=10 ** 8,
                            n_threads=N_THREADS, objectives=["r2", "length"], random_state=0, max_time=OPERON_SECONDS)
    reg.fit(X, y)
    front = [(int(m["length"]), m["tree"]) for m in reg.pareto_front_]
    names_map = {i: n for i, n in enumerate(names)}
    exprs = [(c, reg.get_model_string(mod, 4, names_map), mod) for c, mod in front]
    predict = lambda mod, Z: np.asarray(reg.evaluate_model(mod, Z), float)
    return exprs, predict


def run_pysr(X, y, names):
    from pysr import PySRRegressor
    reg = PySRRegressor(niterations=100000, timeout_in_seconds=PYSR_SECONDS, populations=64, population_size=60,
                        maxsize=45, binary_operators=["+", "-", "*", "/"], unary_operators=["square"],
                        constraints={"/": (-1, 15), "square": 9}, parsimony=1e-4, batching=True, batch_size=2000,
                        parallelism="multithreading", turbo=True, verbosity=0, progress=False,
                        model_selection="best", random_state=0, deterministic=False,
                        output_directory=os.path.join(OUT, "pysr_outputs"))
    reg.fit(X, y, variable_names=names)
    eq = reg.equations_
    exprs = [(int(r.complexity), str(r.equation), i) for i, r in eq.iterrows()]
    predict = lambda i, Z: np.asarray(reg.predict(Z, index=i), float)
    return exprs, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["operon", "pysr", "all"], default="all")
    ap.add_argument("--range", type=float, default=None)
    a = ap.parse_args()
    engines = ["operon", "pysr"] if a.engine == "all" else [a.engine]
    span = a.range
    os.makedirs(OUT, exist_ok=True)
    for engine in engines:
        for cname in CLASSES:
            data, names, _ = class_data(cname, span)
            targets = ("sin", "cos") if span is None else (f"r{span:g}_p",)
            for j, tname in enumerate(targets):
                path = os.path.join(OUT, f"{engine}_{cname}_{tname}.npz")
                if os.path.exists(path):
                    continue
                t = time.time()
                X, y = data["fit"][0], data["fit"][1][:, j]
                exprs, predict = (run_operon if engine == "operon" else run_pysr)(X, y, names)
                comp = np.array([c for c, _, _ in exprs])
                es = np.array([predict(m, data["es"][0]) for _, _, m in exprs])
                va = np.array([predict(m, data["va"][0]) for _, _, m in exprs])
                np.savez(path, complexity=comp, expression=np.array([e for _, e, _ in exprs]),
                         pred_es=es.astype(np.float32), pred_va=va.astype(np.float32),
                         true_es=data["es"][1][:, j], true_va=data["va"][1][:, j], names=names)
                r2 = 1 - ((es - data["es"][1][:, j]) ** 2).mean(1) / data["es"][1][:, j].var()
                print(f"{engine} {cname} {tname}: {len(exprs)} expressions, best es R2 {r2.max():.4f} "
                      f"({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
