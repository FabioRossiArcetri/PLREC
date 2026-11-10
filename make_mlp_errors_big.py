"""MLP slide figure: per-piston held-out error of the 4x256 net, clean single wavelength."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ml_reconstruct as ml

DATA = os.path.join("data", "ml", "pl19_1200")
OUT = os.path.join("Presentation", "Slides", "figures", "slides")
SPAN = 600.0

ds = ml.LanternDataset.load(os.path.join(DATA, "single_ao0.npz"))
S_tr, C_tr = ds.train
S_va, C_va = ds.val
xs = ml.Standardizer().fit(S_tr)
ys = ml.PhasorCodec(2 * SPAN)
params, layers, extra = ml.load_mlp(os.path.join(DATA, "cache_big", "ao0-mlp-256x4.npz"))
g = ys.inverse(ml.mlp_forward(params, xs.transform(S_va)))
per = np.asarray(ml.rmse_nm(g, C_va, 2 * SPAN))
best = int(extra["best_epoch"])
print("per piston:", np.round(per, 2), "mean", per.mean())

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, ax = plt.subplots(figsize=(4.0, 2.55))
b = ax.bar(np.arange(5), per, 0.62, color="#e377c2", edgecolor="#a03c78", lw=0.5)
ax.axhline(per.mean(), color="#333333", ls="--", lw=0.8,
           label=f"mean {per.mean():.2f} nm")
for i, v in enumerate(per):
    ax.text(i, v + 0.06, f"{v:.2f}", ha="center", va="bottom", fontsize=6)
ax.set_xticks(np.arange(5))
ax.set_xticklabels([f"piston {i}" for i in range(5)])
ax.set_ylabel("held-out RMSE modulo $\\lambda$ (nm)")
ax.set_ylim(0, max(per) * 1.30)
ax.grid(alpha=0.3, axis="y", lw=0.3)
ax.legend(loc="upper right", handlelength=1.2, borderpad=0.25)
ax.set_title(f"4 $\\times$ 256 MLP, noiseless: {per.mean():.2f} nm", fontsize=7.5)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_errors_big.pdf"))
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_errors_big.png"), dpi=220)
print(f"written pl19_1200_mlp_errors_big.pdf (best epoch {best})")
