"""Slide figure: the wider MLP on the single-wavelength case (curves + RMSE bars)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ml_reconstruct as ml

DATA = os.path.join("data", "ml", "pl19_1200")
CACHE = os.path.join(DATA, "cache_big")
OUT = os.path.join("Presentation", "Slides", "figures", "slides")
SPAN = 600.0
LEVELS = (0, 30, 60, 90, 120, 150)
OLD = {0: 9.44, 30: 27.6, 60: 48.0, 90: 79.8, 120: 60.5, 150: 66.9}
C_OLD, C_NEW = "#b8c4cf", "#e377c2"          # grey-blue, MLP pink (canonical C6)

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
    rows.append((r, rmse, np.asarray(extra["val"], float)))
    print(f"{r:3d}: {rmse:6.2f} nm")

plt.rcParams.update({"font.size": 6.5, "axes.titlesize": 7.0, "axes.labelsize": 6.5,
                     "legend.fontsize": 5.5, "xtick.labelsize": 6, "ytick.labelsize": 6})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.15, 2.15),
                               gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.32})

for r, rmse, hist in rows:
    ax1.semilogy(np.arange(1, len(hist) + 1), hist, lw=0.7, label=f"{r} nm")
ax1.set_xlabel("epoch")
ax1.set_ylabel("validation loss")
ax1.set_title("4 $\\times$ 256 MLP: held-out loss per level")
ax1.grid(alpha=0.3, which="both", lw=0.3)
ax1.legend(ncol=2, handlelength=1.2, columnspacing=0.8, borderpad=0.2, loc="upper right")

x = np.arange(len(rows))
ax2.bar(x - 0.2, [OLD[r] for r, _, _ in rows], 0.4, color=C_OLD, label="3 $\\times$ 128")
ax2.bar(x + 0.2, [v for _, v, _ in rows], 0.4, color=C_NEW, label="4 $\\times$ 256")
for xi, (_, v, _) in zip(x, rows):
    ax2.text(xi + 0.2, v * 1.06, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5)
ax2.set_xticks(x)
ax2.set_xticklabels([f"{r}" for r, _, _ in rows])
ax2.set_xlabel("wavefront error besides the pistons (nm)")
ax2.set_ylabel("validation RMSE (nm)")
ax2.set_title("single wavelength, 19 ports at 1200 nm")
ax2.set_ylim(0, 88)
ax2.grid(alpha=0.3, axis="y", lw=0.3)
ax2.legend(loc="upper left", handlelength=1.2, borderpad=0.2)

fig.tight_layout(pad=0.25)
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_big_single.pdf"))
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_big_single.png"), dpi=220)
print("written:", os.path.join(OUT, "pl19_1200_mlp_big_single.pdf"))
