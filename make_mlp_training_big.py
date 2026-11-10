"""MLP slide figure: the old 3x128 run and the new 4x256 run, train and validation."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import ml_reconstruct as ml

DATA = os.path.join("data", "ml", "pl19_1200")
OUT = os.path.join("Presentation", "Slides", "figures", "slides")
OLD = os.path.join(DATA, "cache_single_phasor", "mlp-19x128x128x128x10-e500.npz")

d_old = np.load(OLD, allow_pickle=True)
old_tr, old_va = np.asarray(d_old["extra_train"], float), np.asarray(d_old["extra_val"], float)

params, layers, extra = ml.load_mlp(os.path.join(DATA, "cache_big", "ao0-mlp-256x4.npz"))
new_tr, new_va = np.asarray(extra["train"], float), np.asarray(extra["val"], float)
best = int(extra["best_epoch"])
print(f"old: {len(old_tr)} epochs, val {old_va.min():.2e}; "
      f"new: {len(new_tr)} epochs, val {new_va.min():.2e} at {best}")

plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "legend.fontsize": 5.8,
                     "xtick.labelsize": 6.5, "ytick.labelsize": 6.5})
fig, ax = plt.subplots(figsize=(4.05, 2.9))
ax.semilogy(np.arange(1, len(old_tr) + 1), old_tr, color="#b8c4cf", lw=0.8,
            label="3 $\\times$ 128, minibatch (dropout on)")
ax.semilogy(np.arange(1, len(old_va) + 1), old_va, color="#4c72b0", lw=0.9,
            label="3 $\\times$ 128, held out")
ax.semilogy(np.arange(1, len(new_tr) + 1), new_tr, color="#f2c3dd", lw=0.8,
            label="4 $\\times$ 256, minibatch (dropout on)")
ax.semilogy(np.arange(1, len(new_va) + 1), new_va, color="#e377c2", lw=1.1,
            label="4 $\\times$ 256, held out")
ax.axvline(best, color="#a03c78", ls=":", lw=0.8)
ax.annotate(f"best epoch {best}", xy=(best, new_va.min()),
            xytext=(best * 0.55, new_va.min() * 0.40), fontsize=5.8, color="#a03c78",
            arrowprops=dict(arrowstyle="-", color="#a03c78", lw=0.5))
ax.set_xlabel("epoch")
ax.set_ylabel("loss (standardised phasors)")
ax.set_xlim(0, len(new_va) * 1.02)
ax.grid(alpha=0.3, which="both", lw=0.3)
ax.legend(loc="upper right", handlelength=1.4, borderpad=0.25, labelspacing=0.25)
ax.set_title("clean, single wavelength: 4.6 nm held out (was 9.4 nm)", fontsize=7)
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_training_big.pdf"))
fig.savefig(os.path.join(OUT, "pl19_1200_mlp_training_big.png"), dpi=220)
print("written pl19_1200_mlp_training_big.pdf")
