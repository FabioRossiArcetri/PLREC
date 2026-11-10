"""Continue the big-MLP models from their saved weights until the held-out loss settles.

Adds to scripts/train_mlp_big.py: --continue resumes the model already saved for that job
(init_params + appended history), trains up to --target total epochs, and stops when the
held-out loss has not improved for PATIENCE epochs -- the rule the first campaign never
reached because it hit its 6000-epoch cap.

    python scripts/continue_mlp_big.py --stage single --level 30
    python scripts/continue_mlp_big.py --stage perlambda --wl 7
    python scripts/continue_mlp_big.py --list
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
LEVELS = (0, 30, 60, 90, 120, 150)
TARGET = int(os.environ.get("SR_TARGET", "20000"))
PATIENCE = int(os.environ.get("SR_PATIENCE", "500"))
LR = float(os.environ.get("SR_LR", "3e-4"))
SPAN_DEFAULT = 600.0


def resume(tag, path, Xtr, Ytr, Xva, Yva):
    if not os.path.exists(path):
        raise SystemExit(f"{path}: no saved model to continue")
    params, layers, extra = ml.load_mlp(path)
    have = len(np.asarray(extra["train"], float))
    print(f"{tag}: continuing from epoch {have}", flush=True)
    params, hist = ml.train_mlp(LAYERS, Xtr, Ytr, Xva, Yva, epochs=TARGET,
                                learning_rate=LR, log_every=1000,
                                init_params=params,
                                history={"train": np.asarray(extra["train"], float).tolist(),
                                         "val": np.asarray(extra["val"], float).tolist()},
                                eval_train=False)
    val = np.asarray(hist["val"], float)
    best = int(np.argmin(val)) + 1
    stop = len(val)
    tail = val[best - 1:best - 1 + PATIENCE]
    improved = bool(len(tail) == PATIENCE and np.min(tail[1:]) < tail[0])
    print(f"{tag}: epochs {have} -> {stop}, best {best} ({val[best - 1]:.4e}), "
          f"patience satisfied: {not improved}", flush=True)
    ml.save_mlp(path, params, LAYERS, train=hist["train"], val=hist["val"], best_epoch=best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["single", "perlambda"], required=True)
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--wl", type=int, default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for r in LEVELS:
            print(f"single {r}")
        for w in range(21):
            print(f"perlambda {w}")
        return

    if a.stage == "single":
        r = a.level
        ds = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{r}.npz"))
        xs = ml.Standardizer().fit(ds.train[0])
        ys = ml.PhasorCodec(2 * SPAN_DEFAULT)
        resume(f"ao{r}", os.path.join(OUT, f"ao{r}-mlp-256x4.npz"),
               xs.transform(ds.train[0]), ys.transform(ds.train[1]),
               xs.transform(ds.val[0]), ys.transform(ds.val[1]))
    else:
        ds = ml.LanternDataset.load(os.path.join(DATA, "spectral_ao120_21wl.npz"))
        wls = np.asarray(ds.wavelengths_nm, float)
        S = np.asarray(ds.signals)
        P = np.concatenate([np.asarray(ds.train[1]), np.asarray(ds.val[1])], 0)
        n_tr = len(ds.train[0])
        w = a.wl
        lam = float(wls[w])
        xs = ml.Standardizer().fit(S[:n_tr, w, :])
        ys = ml.PhasorCodec(lam)
        resume(f"w{w} ({lam:.0f} nm)", os.path.join(OUT, f"w{w}-mlp-256x4.npz"),
               xs.transform(S[:n_tr, w, :]), ys.transform(P[:n_tr]),
               xs.transform(S[n_tr:, w, :]), ys.transform(P[n_tr:]))


if __name__ == "__main__":
    main()
