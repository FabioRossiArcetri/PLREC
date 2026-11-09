"""A lantern's single-wavelength reconstruction at a given physical piston range
(nm), scored exactly as trainMLP.ipynb does (linear ridge, 5-NN, MLP 3 x 128 /
500 epochs / Adam 3e-4, 8000 train / 2000 validation).

Compares lanterns at the same piston range in nm -- the quantity a telescope
sets -- rather than at the same fraction of their wavelength.

    python scripts/same_range.py <lantern> <span_nm>     # e.g. pl19_1200 400
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
LANTERN, SPAN = sys.argv[1], float(sys.argv[2])
os.environ["CBEAM_LANTERN"] = LANTERN
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import ml_reconstruct as ml                                            # noqa: E402
from ml_reconstruct.simulate import make_single_dataset                # noqa: E402
from sklearn.linear_model import Ridge                                 # noqa: E402
from sklearn.neighbors import KNeighborsRegressor                      # noqa: E402

out = os.path.join(ROOT, "data", "ml", "same_range", f"{LANTERN}_single_{SPAN:g}nm.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
t = time.time()
if os.path.exists(out):
    ds = ml.LanternDataset.load(out)
else:
    ds = make_single_dataset(10000, 8000, SPAN, lantern=LANTERN, prop_chunk_size=1000)
    ds.save(out)
print(f"{LANTERN} at +-{SPAN:g} nm: dataset {time.time() - t:.0f} s", flush=True)

S_tr, C_tr = ds.train
S_va, C_va = ds.val
xs, ys = ml.Standardizer().fit(S_tr), ml.Standardizer().fit(C_tr)
Xtr, Xva, Ytr = xs.transform(S_tr), xs.transform(S_va), ys.transform(C_tr)


def report(name, Yhat):
    C = ys.inverse(np.asarray(Yhat))
    per = np.sqrt(np.mean((C - C_va) ** 2, axis=0))
    print(f"  {name:8s} mean {per.mean():7.1f} nm   per piston {np.round(per, 1)}", flush=True)


report("linear", Ridge(1.0).fit(Xtr, Ytr).predict(Xva))
report("kNN", KNeighborsRegressor(5, weights="distance").fit(Xtr, Ytr).predict(Xva))
layers = [Xtr.shape[1], 128, 128, 128, C_tr.shape[1]]
params, hist = ml.train_mlp(layers, Xtr, Ytr, Xva, ys.transform(C_va), epochs=500, learning_rate=3e-4)
report("MLP", ml.mlp_forward(params, Xva))
print(f"done in {time.time() - t:.0f} s")
