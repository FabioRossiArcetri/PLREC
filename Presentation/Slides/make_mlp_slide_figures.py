#!/usr/bin/env python
"""The two single-wavelength figures of the MLP / comparison slides, from data (19 ports, 1200 nm):

  pl19_1200_mlp_new                 the 4x256 MLP (cache_big/ao0): training history (minibatch and
                                    held-out loss, best epoch) and truth vs prediction on the
                                    independent noiseless test set (single_ao0_test.npz)
  pl19_1200_single_comparison_new   every method's RMSE modulo lambda on that test set, sorted
                                    (left), against degrees of freedom per piston (right)

Test-set scores from scripts/score_test.py (data/ml/pl19_1200/test_scores.npz, level 0);
symbolic regression keeps its validation value (not re-scored). Colours: slidefig.PALETTE.
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

DATA = os.path.join(ROOT, "data", "ml", "pl19_1200")
LAM = 1200.0
SYMBOLIC_VAL = 91.3                       # validation RMSE of the symbolic reconstructor (not re-scored)
P = slidefig.PALETTE
DOF = slidefig.DOF["pl19_1200"]["single"] | {"mlp": 41013}   # 205 066 weights and biases / 5 pistons
plt.rcParams.update(slidefig.RC)

# ------------------------------------------------------------------ the MLP figure
tr = ml.LanternDataset.load(os.path.join(DATA, "single_ao0.npz"))
te = ml.LanternDataset.load(os.path.join(DATA, "single_ao0_test.npz"))
xs, ys = ml.Standardizer().fit(tr.train[0]), ml.PhasorCodec(LAM)
params, layers, extra = ml.load_mlp(os.path.join(DATA, "cache_big", "ao0-mlp-256x4.npz"))
C_te = te.coeff[:, :5]
C_hat = ys.inverse(np.asarray(ml.mlp_forward(params, xs.transform(te.signals))))
err = ml.rmse_nm(C_hat, C_te, LAM)
shown = C_te + ml.wrap(C_hat - C_te, LAM)            # an estimate across the wrap drawn at the truth
fig, ax = plt.subplots(1, 2, figsize=(7.0 * slidefig.CM, 3.6 * slidefig.CM), layout="constrained")
ep = np.arange(1, len(extra["train"]) + 1)
ax[0].semilogy(ep, extra["train"], color=P["mlp"], alpha=0.35, lw=0.6, label="minibatch")
ax[0].semilogy(ep, extra["val"], color=P["mlp"], lw=0.8, label="held out")
ax[0].axvline(int(extra["best_epoch"]), color="0.3", ls=":", lw=0.6)
ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss")
ax[0].set_title(f"4$\\times$256: best epoch {int(extra['best_epoch'])}")
ax[0].legend(loc="upper right"); ax[0].grid(alpha=0.3, which="both")
ax[1].plot(C_te.ravel(), shown.ravel(), ".", ms=0.8, color=P["mlp"], alpha=0.4)
ax[1].plot([-600, 600], [-600, 600], "--", color="0.2", lw=0.6)
ax[1].set_xlim(-600, 600); ax[1].set_ylim(-600, 600)
ax[1].set_xlabel("true piston (nm)"); ax[1].set_ylabel("predicted (nm)")
ax[1].set_title(f"test set: {err.mean():.2f} nm"); ax[1].grid(alpha=0.3)
slidefig.export(fig, "pl19_1200_mlp_new", native=True)
print("MLP test RMSE per piston", np.round(err, 2), "mean", round(float(err.mean()), 2))

# ------------------------------------------------------------------ the comparison figure
sc = np.load(os.path.join(DATA, "test_scores.npz"), allow_pickle=True)["single"][0]
rows = [("linear", "linear", sc["linear"][0][1]), ("symb.", "symbolic", SYMBOLIC_VAL),
        ("d2", "dense d2", sc["dense poly, degree 2"][0][1]), ("d3", "dense d3", sc["dense poly, degree 3"][0][1]),
        ("OMP", "omp", sc["omp K=1282"][0][1]), ("c.f.", "closed form", sc["closed form"][0][1]),
        ("MLP", "mlp", sc["MLP 4x256"][0][1])]
rows.sort(key=lambda r: -r[2])
fig, ax = plt.subplots(1, 2, figsize=(13.0 * slidefig.CM, 4.0 * slidefig.CM), layout="constrained")
x = np.arange(len(rows))
ax[0].bar(x, [r[2] for r in rows], color=[P[r[1]] for r in rows], width=0.7)
for i, r in enumerate(rows):
    ax[0].text(i, r[2] * 1.6, f"{r[2]:.3g}" if r[2] >= 0.01 else "<0.01", ha="center", va="bottom")
ax[0].set_yscale("log"); ax[0].set_ylim(top=1e4)
ax[0].set_xticks(x, [r[0] for r in rows]); ax[0].set_ylabel("RMSE (nm)")
ax[0].set_title("by decreasing error (test set)"); ax[0].grid(alpha=0.3, axis="y", which="both")
for i, r in enumerate(rows):
    ax[1].loglog(DOF[r[1]], r[2], "o", color=P[r[1]], ms=3.5)
    ax[1].annotate(r[0], (DOF[r[1]], r[2]), xytext=(4, 3 if i % 2 else -8), textcoords="offset points", color=P[r[1]])
ax[1].set_xlabel("degrees of freedom per piston"); ax[1].set_ylabel("RMSE (nm)")
ax[1].set_title("accuracy against degrees of freedom"); ax[1].grid(alpha=0.3, which="both")
slidefig.export(fig, "pl19_1200_single_comparison_new", native=True)
print("comparison:", [(r[0], round(float(r[2]), 4)) for r in rows])
