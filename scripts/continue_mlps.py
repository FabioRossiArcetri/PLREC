"""Continue training the MLP models already saved in a lantern's caches.

The saved models were stopped at 500 epochs while the validation loss was still
falling (``best_epoch`` equals the last epoch). This script reloads each one and
trains it further with the same data and the same preprocessing, appending the
new epochs to the stored history so the loss curve stays continuous.

    python scripts/continue_mlps.py --lantern pl19_1200 --epochs 500
    python scripts/continue_mlps.py --lantern pl19_1200 --epochs 50 --dry-run

Two kinds of model are handled:
  * single wavelength, one per wavefront-error level:  cache_single_phasor/ao<rms>-mlp-*.npz
  * the clean single-wavelength model of the notebook:  cache_single_phasor/mlp-*.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ml_reconstruct as ml                                     # noqa: E402
from ml_reconstruct.presets import settings                     # noqa: E402


def load_arrays(path, codec=None):
    """Signals and targets of a dataset, standardised exactly as the notebook does."""
    ds = ml.LanternDataset.load(path)
    S_tr, C_tr = ds.train
    S_va, C_va = ds.val
    standardiser = ml.Standardizer().fit(S_tr)
    if codec is None:
        codec = ml.PhasorCodec(2 * np.abs(C_tr).max())
    return (standardiser.transform(S_tr), standardiser.transform(S_va),
            codec.transform(C_tr), codec.transform(C_va))


def continue_one(path, Xtr, Ytr, Xva, Yva, epochs, lr, dry_run=False):
    params, layers, extra = ml.load_mlp(path)
    hist = {"train": list(extra["train"]), "val": list(extra["val"]),
            "best_epoch": int(extra["best_epoch"])}
    n_before = len(hist["val"])
    # the like-for-like training loss, without dropout, before continuing
    train_eval_before = float(ml.huber_loss(np.asarray(Ytr),
                                            np.asarray(ml.mlp_forward(params, np.asarray(Xtr)))))
    print(f"\n{os.path.basename(path)}")
    print(f"  stored history: {n_before} epochs, best {hist['best_epoch']}, "
          f"val {hist['val'][-1]:.4e}, train(minibatch, dropout) {hist['train'][-1]:.4e}")
    print(f"  training set without dropout: {train_eval_before:.4e}"
          f"   (validation {hist['val'][-1]:.4e})")
    if dry_run:
        return None
    best, hist = ml.train_mlp(layers, Xtr, Ytr, Xva, Yva, epochs=epochs, learning_rate=lr,
                              init_params=params, history=hist, eval_train=True, log_every=50)
    ml.save_mlp(path, best, layers, train=hist["train"], val=hist["val"],
                best_epoch=hist["best_epoch"], train_eval=hist.get("train_eval", []))
    rmse_before = float(ml.rmse_nm(np.asarray(ml.mlp_forward(params, np.asarray(Xva))),
                                   np.asarray(Yva), LAMBDA_CODE[0]).mean())
    rmse_after = float(ml.rmse_nm(np.asarray(ml.mlp_forward(best, np.asarray(Xva))),
                                  np.asarray(Yva), LAMBDA_CODE[0]).mean())
    print(f"  now {len(hist['val'])} epochs, best {hist['best_epoch']}, "
          f"val {hist['val'][-1]:.4e}  (was {hist['val'][n_before - 1]:.4e})")
    print(f"  validation RMSE {rmse_before:.2f} -> {rmse_after:.2f} nm")
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lantern", default="pl19_1200")
    ap.add_argument("--epochs", type=int, default=500, help="epochs to add")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--levels", default="", help="comma separated rms levels, default all cached")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    S = settings(args.lantern)
    DATA = S["data_dir"]
    global LAMBDA_CODE
    codec = ml.PhasorCodec(2 * S["single_span_nm"])
    LAMBDA_CODE = [2 * S["single_span_nm"]]

    kind = os.environ.get("WFE_PSD", "ao")
    paths = sorted(glob.glob(os.path.join(DATA, "cache_single_phasor", "*-mlp-*.npz")))
    if args.levels:
        want = {f"ao{r}" for r in args.levels.split(",")}
        paths = [p for p in paths if os.path.basename(p).split("-mlp")[0] in want]
    print(f"lantern {args.lantern}: {len(paths)} MLP models")

    for path in paths:
        tag = os.path.basename(path).split("-mlp")[0]          # 'ao120' or 'mlp'
        level = tag[2:] if tag.startswith("ao") else "0"
        ds_path = (f"{DATA}/single_{kind}{level}.npz" if tag.startswith("ao")
                   else f"{DATA}/{os.path.basename(S['single'])}")
        if not os.path.exists(ds_path):
            print(f"  -- {tag}: dataset {ds_path} missing, skipped")
            continue
        Xtr, Xva, Ytr, Yva = load_arrays(ds_path, codec=codec)
        continue_one(path, Xtr, Ytr, Xva, Yva, args.epochs, args.lr, args.dry_run)


LAMBDA_CODE = [0.0]

if __name__ == "__main__":
    main()
