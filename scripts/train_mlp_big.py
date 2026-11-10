"""Train the bigger MLP family: 4 hidden layers of 256, early stopping on the held-out loss.

Stage "single": the six wavefront-error levels of the single-wavelength dataset
(19 -> 4x256 -> 10), one model per level.
Stage "perlambda": the 21 per-wavelength models of the dispersed dataset (19 -> 4x256 -> 10
at each wavelength, phasor targets with that wavelength's period).

Training stops when the held-out loss has not improved for PATIENCE epochs; the weights of
the best epoch are kept and the whole history is stored. Outputs do not touch the existing
caches: they go to cache_big/ with the layer list in the name.

    python scripts/train_mlp_big.py --stage single
    python scripts/train_mlp_big.py --stage perlambda --wl 7
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml                        # noqa: E402

DATA = os.path.join(ROOT, "data", "ml", "pl19_1200")
OUT = os.path.join(DATA, "cache_big")
LAYERS = [19, 256, 256, 256, 256, 10]
PATIENCE = 300
MAX_EPOCHS = 6000
LR = 3e-4
SPAN_NM = 600.0
LEVELS = (0, 30, 60, 90, 120, 150)


def train(Xtr, Ytr, Xva, Yva, tag, path):
    print(f"\n{tag}: {LAYERS}, patience {PATIENCE}, max {MAX_EPOCHS} epochs", flush=True)
    params, hist = ml.train_mlp(LAYERS, Xtr, Ytr, Xva, Yva, epochs=MAX_EPOCHS, learning_rate=LR,
                                log_every=250)
    val = np.asarray(hist["val"])
    best = int(hist["best_epoch"])
    print(f"  best epoch {best} of {len(val)}, val {val[best - 1]:.4e}; "
          f"last {val[-1]:.4e}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ml.save_mlp(path, params, LAYERS, train=hist["train"], val=hist["val"],
                best_epoch=best)
    print(f"  saved {path}", flush=True)


def stage_single(level=None):
    for r in ([level] if level is not None else LEVELS):
        ds = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{r}.npz"))
        S_tr, C_tr = ds.train
        S_va, C_va = ds.val
        xs, ys = ml.Standardizer().fit(S_tr), ml.PhasorCodec(2 * SPAN_NM)
        rmse = lambda p: float(ml.rmse_nm(np.asarray(ml.mlp_forward(p, xs.transform(S_va))),
                                          ys.transform(C_va), 2 * SPAN_NM).mean())
        tag = f"ao{r}"
        train(xs.transform(S_tr), ys.transform(C_tr), xs.transform(S_va), ys.transform(C_va),
              tag, os.path.join(OUT, f"{tag}-mlp-256x4.npz"))


def stage_perlambda(wl):
    ds = ml.LanternDataset.load(os.path.join(DATA, "spectral_ao120_21wl.npz"))
    wls = np.asarray(ds.wavelengths_nm, float)
    S = np.asarray(ds.signals)
    P = np.concatenate([np.asarray(ds.train[1]), np.asarray(ds.val[1])], 0)
    n_tr = len(ds.train[0])
    for w in (range(len(wls)) if wl is None else [wl]):
        lam = float(wls[w])
        Xall = S[:, w, :]
        xs, ys = ml.Standardizer().fit(Xall[:n_tr]), ml.PhasorCodec(lam)
        tag = f"w{w}"
        train(xs.transform(Xall[:n_tr]), ys.transform(P[:n_tr]),
              xs.transform(Xall[n_tr:]), ys.transform(P[n_tr:]),
              f"{tag} ({lam:.0f} nm)", os.path.join(OUT, f"{tag}-mlp-256x4.npz"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["single", "perlambda"], required=True)
    ap.add_argument("--wl", type=int, default=None)
    ap.add_argument("--level", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.stage == "single":
        stage_single(a.level)
    else:
        stage_perlambda(a.wl)


if __name__ == "__main__":
    main()
