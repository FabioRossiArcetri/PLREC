"""Look at the big-MLP single-wavelength models: training curves and validation RMSE."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ml_reconstruct as ml

DATA = os.path.join("data", "ml", "pl19_1200")
CACHE = os.path.join(DATA, "cache_big")
SPAN = 600.0
LEVELS = (0, 30, 60, 90, 120, 150)
OLD = {0: 9.44, 30: 27.6, 60: 48.0, 90: 79.8, 120: 60.5, 150: 66.9}   # 3x128, nm

rows = []
for r in LEVELS:
    ds = ml.LanternDataset.load(os.path.join(DATA, f"single_ao{r}.npz"))
    S_tr, C_tr = ds.train
    S_va, C_va = ds.val
    xs = ml.Standardizer().fit(S_tr)
    ys = ml.PhasorCodec(2 * SPAN)
    params, layers, extra = ml.load_mlp(os.path.join(CACHE, f"ao{r}-mlp-256x4.npz"))
    g = ys.inverse(ml.mlp_forward(params, xs.transform(S_va)))
    rmse = float(np.asarray(ml.rmse_nm(g, C_va, 2 * SPAN)).mean())
    hist = np.asarray(extra["val"], float)
    best = int(extra["best_epoch"])
    rows.append((r, rmse, hist, best))
    print(f"{r:3d} nm: val RMSE {rmse:7.2f} nm  (3x128: {OLD[r]:5.1f})  "
          f"epochs {len(hist)}, best {best}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
for r, rmse, hist, best in rows:
    ax1.semilogy(np.arange(1, len(hist) + 1), hist, lw=1.0, label=f"{r} nm")
ax1.set_xlabel("epoch")
ax1.set_ylabel("validation loss (standardised targets)")
ax1.set_title("4 x 256 MLP: validation loss per level")
ax1.grid(alpha=0.3, which="both")
ax1.legend(fontsize=7, ncol=2)

levels = [r for r, *_ in rows]
new = [rm for _, rm, _, _ in rows]
old = [OLD[r] for r in levels]
x = np.arange(len(levels))
ax2.bar(x - 0.2, old, 0.4, label="3 x 128 (before)", color="#9ecae1")
ax2.bar(x + 0.2, new, 0.4, label="4 x 256 (now)", color="#ff7f0e")
for xi, v in zip(x, new):
    ax2.text(xi + 0.2, v * 1.05, f"{v:.1f}", ha="center", fontsize=7)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{r}" for r in levels])
ax2.set_xlabel("wavefront error besides the pistons (nm)")
ax2.set_ylabel("validation RMSE (nm)")
ax2.set_title("single wavelength, 19 ports at 1200 nm")
ax2.grid(alpha=0.3, axis="y")
ax2.legend(fontsize=8)
fig.tight_layout()
fig.savefig("mlp_big_single.png", dpi=170)
print("wrote mlp_big_single.png")
