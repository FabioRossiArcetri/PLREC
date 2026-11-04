"""Does the easy piston follow the reference petal? (piston2.ipynb)

The same 800 nm signals (single_800nm_50k.npz), with the pistons expressed
relative to each of the 6 petals in turn (ml_reconstruct.petals.rereference:
the physical configurations, and so the signals, are the same). One 4 x 256
MLP per reference, trained as in scripts/mlp_hpo.py (cosine from 3e-4, 2000
epochs) on the first N_FIT rows, selected on rows 36000-40000 and scored on
the 10000 validation rows. If the easy piston is the one opposite the
reference, its ease comes from the pupil's twin symmetry; if it stays on one
petal whatever the reference, from the lantern (e.g. its clocking).

    python scripts/piston2_reref.py      (-> data/ml/piston2/reref_<petal>.npz)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml                     # noqa: E402
from ml_reconstruct import petals               # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "piston2")
DATASET = os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")
REF, N_PETALS = 5, 6                            # the dataset's reference petal (none of modes 0-4)
N_FIT, ES = 18000, slice(36000, 40000)


def main():
    os.makedirs(OUT, exist_ok=True)
    ds = ml.LanternDataset.load(DATASET)
    S, VA = ds.signals, slice(ds.n_train, None)
    xs = ml.Standardizer().fit(S[:N_FIT])
    X_fit, X_es, X_va = (xs.transform(S[r]) for r in (slice(0, N_FIT), ES, VA))
    for new_ref in range(N_PETALS):
        path = os.path.join(OUT, f"reref_{new_ref}.npz")
        if os.path.exists(path):
            continue
        C = petals.rereference(ds.coeff, REF, new_ref)          # (n, 5), petals != new_ref
        ys = ml.Standardizer().fit(C[:N_FIT])
        params, hist = ml.train_mlp([19, 256, 256, 256, 256, 5], X_fit, ys.transform(C[:N_FIT]),
                                    X_es, ys.transform(C[ES]), epochs=2000, learning_rate=3e-4,
                                    lr_schedule="cosine", init_seed=0, train_seed=100, log_every=0)
        pred = ys.inverse(np.asarray(ml.mlp_forward(params, X_va)))
        petal_ids = [k for k in range(N_PETALS) if k != new_ref]
        np.savez(path, pred_va=pred, true_va=C[VA], petal_ids=petal_ids, best_epoch=hist["best_epoch"])
        print(f"reference petal {new_ref}: RMSE per petal {dict(zip(petal_ids, ml.rmse_nm(pred, C[VA]).round(2)))}",
              flush=True)


if __name__ == "__main__":
    main()
