"""Template symbolic regression of the single-wavelength piston (symbolic.ipynb):
PySR searching expressions of the form

    theta = atan2(f(features), g(features)),      p = theta lambda / (2 pi),

with a periodic loss 1 - cos(theta_hat - theta) on the piston's phase, instead
of fitting its sine and cosine separately (scripts/symbolic_sr.py). Same
features, classes and rows as symbolic_sr.py.

    python scripts/symbolic_template.py --cls opposite [--range R]
      -> data/ml/symbolic/pysr-template_<class>[_r<R>].npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symbolic_sr as base                       # noqa: E402  (data and constants)

SECONDS = 900


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", required=True, choices=list(base.CLASSES))
    ap.add_argument("--range", type=float, default=None)
    a = ap.parse_args()
    tag = f"{a.cls}" + ("" if a.range is None else f"_r{a.range:g}")
    path = os.path.join(base.OUT, f"pysr-template_{tag}.npz")
    if os.path.exists(path):
        return
    data, names, _ = base.class_data(a.cls, a.range)
    to_theta = (lambda T: np.arctan2(T[:, 0], T[:, 1])) if a.range is None else \
        (lambda T: 2 * np.pi * T[:, 0] / base.LAMBDA_NM)
    from pysr import PySRRegressor, TemplateExpressionSpec
    args = ", ".join(names)
    spec = TemplateExpressionSpec(expressions=["f", "g"], variable_names=names,
                                  combine=f"atan(f({args}), g({args}))")
    reg = PySRRegressor(expression_spec=spec, niterations=100000, timeout_in_seconds=SECONDS,
                        populations=64, population_size=60, maxsize=60,
                        binary_operators=["+", "-", "*", "/"], unary_operators=["square"],
                        elementwise_loss="periodic(x, y) = 1 - cos(x - y)", batching=True, batch_size=2000,
                        parallelism="multithreading", turbo=True, verbosity=0, progress=False,
                        model_selection="best", random_state=0, deterministic=False,
                        output_directory=os.path.join(base.OUT, "pysr_outputs"))
    t = time.time()
    X, th = data["fit"][0], to_theta(data["fit"][1])
    reg.fit(X, th)
    eq = reg.equations_
    rows = range(len(eq))
    pred = {s: np.array([np.asarray(reg.predict(data[s][0], index=i), float) for i in rows]) for s in ("es", "va")}
    np.savez(path, complexity=eq["complexity"].to_numpy(), expression=np.array([str(e) for e in eq["equation"]]),
             pred_es=pred["es"], pred_va=pred["va"], true_es=to_theta(data["es"][1]),
             true_va=to_theta(data["va"][1]), names=names)
    lam = base.LAMBDA_NM / (2 * np.pi)
    err = lambda p, t: np.sqrt(np.mean((np.angle(np.exp(1j * (p - t))) * lam) ** 2, axis=-1))
    print(f"template {tag}: {len(eq)} expressions, best held-out piston RMSE "
          f"{err(pred['es'], to_theta(data['es'][1])).min():.2f} nm ({time.time() - t:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
