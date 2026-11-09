#!/usr/bin/env python
"""The port-count ambiguity figure: signal collisions vs. piston range.

Reads figures/sim/<lantern>_collisions.npz written by diag_collisions.py (one
per lantern) and plots the fraction of configurations whose closest-signal
neighbour has pistons more than lambda/4 rms away, for every lantern found.
"""
import glob
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import slidefig

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "figures", "sim")
LABEL = {"pl19_800": "19 ports, 800 nm (reference)", "pl19_1200": "19 ports, 1200 nm",
         "pl7_1200": "7 ports, 1200 nm"}
STYLE = {"pl19_800": ("C0", "o"), "pl19_1200": ("C2", "^"), "pl7_1200": ("C1", "s")}

plt.rcParams.update(slidefig.RC)                  # drawn at its size on the slide (cm)
fig, ax = plt.subplots(figsize=(7.6 * slidefig.CM, 5.0 * slidefig.CM), layout="constrained")
for path in sorted(glob.glob(os.path.join(SIM, "*_collisions.npz"))):
    name = os.path.basename(path).replace("_collisions.npz", "")
    d = np.load(path)
    c, m = STYLE.get(name, ("k", "x"))
    ax.plot(d["fracs"], 100 * np.asarray(d["collision"]), m + "-", color=c, label=LABEL.get(name, name))
ax.set_xlabel(r"piston range  $\pm x\,\lambda$  (each petal, uniform)")
ax.set_ylabel("collisions (%)")
ax.set_xticks([1 / 12, 1 / 6, 1 / 4, 1 / 3, 5 / 12, 1 / 2])
ax.set_xticklabels(["1/12", "1/6", "1/4", "1/3", "5/12", "1/2"])
ax.axvline(0.5, color="gray", ls="--", lw=0.6)        # the study range, +-lambda/2
ax.set_ylim(bottom=0); ax.legend(loc="upper left"); ax.grid(alpha=0.3)
ax.set_title(r"closest-signal neighbour with pistons $> \lambda/4$ rms away, modulo $\lambda$")
slidefig.export(fig, "ambiguity_vs_range", native=True)
print("saved")
