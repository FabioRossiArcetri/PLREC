#!/usr/bin/env python
"""The PSF-zoo figures of the appendix (19 ports, 800 nm), drawn at slide size from data:

  pl19_800_psf_zoo_piston_a / _b   petal 0 at 0 (flat), 50, 100, 200 | 300, 400, 600, 800 nm
  pl19_800_psf_zoo_twins_a / _b    the typical and critical pairs | the point-symmetric pair

One column per configuration, four rows: the pupil OPD; the focal-plane intensity (log, relative
to the flat peak; dashed: the lantern input); the focal-plane field (hue = phase, brightness =
intensity over 4 decades); the 19 core powers on the output layout (fraction of the incident
light). Data: data/ml/psf_zoo/channels_800nm.npz (scripts/psf_zoo.py), the fields recomputed
from the pupil (ml_reconstruct.piston2.focal_fields), as in psf_zoo.ipynb.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import slidefig                                                         # noqa: E402
from ml_reconstruct.psfzoo import strehl, pupil_opd                     # noqa: E402
from ml_reconstruct.piston2 import focal_fields, complex_image          # noqa: E402
from ml_reconstruct.simulate import bp, output_core_xy                  # noqa: E402

zoo = np.load(os.path.join(ROOT, "data", "ml", "psf_zoo", "channels_800nm.npz"))
names, C, S = list(zoo["names"]), zoo["coeff"], zoo["signals"]
p = bp.get_simulation_parameters(3, 0.8)
ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
E, ext = focal_fields(ifunc, p, C)
opd = pupil_opd(ifunc, C)
E_flat = E[names.index("flat wavefront")]
I0 = np.abs(E_flat).max() ** 2
SR = strehl(E, E_flat)
XY = output_core_xy(p)
R_CLAD = 9.0
plt.rcParams.update(slidefig.RC)

LABEL = {"flat wavefront": "flat", "typical: p": "typical $p$", "typical: twin Tp": "its twin $Tp$",
         "critical: p": "critical $p$", "critical: twin Tp": "its twin $Tp$",
         "point-symmetric: p": "point-symmetric $p$", "point-symmetric: -p (= its twin)": "$-p$ (= its twin)"}


def label(n):
    return LABEL.get(n, n.replace("petal 0 at ", "petal 0: "))


def draw(cols, name, opd_lim, w_cm):
    n = len(cols)
    idx = [names.index(c) for c in cols]
    fig, ax = plt.subplots(4, n, figsize=(w_cm * slidefig.CM, 6.3 * slidefig.CM), layout="constrained",
                           squeeze=False)
    smax = max(S[i].max() for i in idx)
    for j, i in enumerate(idx):
        a = ax[0, j]
        im0 = a.imshow(opd[i], cmap="RdBu_r", vmin=-opd_lim, vmax=opd_lim)
        a.set_title(label(names[i]))
        a = ax[1, j]
        im1 = a.imshow(np.log10(np.abs(E[i]) ** 2 / I0 + 1e-6), extent=ext, origin="lower", cmap="inferno",
                       vmin=-4, vmax=0)
        a.text(0.03, 0.97, f"Strehl {SR[i]:.2f}", transform=a.transAxes, va="top", color="w", fontsize=4.6)
        a = ax[2, j]
        a.imshow(complex_image(E[i]), extent=ext, origin="lower")
        for a in (ax[1, j], ax[2, j]):
            a.add_patch(plt.Circle((0, 0), R_CLAD, fill=False, ec="c", lw=0.5, ls="--"))
        a = ax[3, j]
        sc = a.scatter(XY[:, 0], XY[:, 1], c=S[i], s=16, cmap="viridis", vmin=0, vmax=smax,
                       edgecolors="0.3", linewidths=0.3)
        a.set_aspect("equal"); a.margins(0.15)
        a.text(0.5, -0.02, f"coupled {S[i].sum():.0%}", transform=a.transAxes, ha="center", va="top",
               fontsize=4.6)
        for r in range(4):
            ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            for sp in ax[r, j].spines.values():
                sp.set_visible(r != 0 and r != 3)
    ax[0, 0].set_ylabel("pupil OPD"); ax[1, 0].set_ylabel("PSF (log)"); ax[2, 0].set_ylabel("field")
    ax[3, 0].set_ylabel("19 cores")
    fig.colorbar(im0, ax=ax[0, :], fraction=0.03, pad=0.01, label="nm")
    fig.colorbar(im1, ax=ax[1, :], fraction=0.03, pad=0.01, label="log$_{10}$ I/I$_0$")
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(-np.pi, np.pi))
    cb = fig.colorbar(sm, ax=ax[2, :], fraction=0.03, pad=0.01, ticks=[-np.pi, 0, np.pi], label="phase")
    cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
    fig.colorbar(sc, ax=ax[3, :], fraction=0.03, pad=0.01, label="power")
    slidefig.export(fig, name, native=True)


draw(["flat wavefront", "petal 0 at 50 nm", "petal 0 at 100 nm", "petal 0 at 200 nm"],
     "pl19_800_psf_zoo_piston_a", 800, 8.6)
draw(["petal 0 at 300 nm", "petal 0 at 400 nm", "petal 0 at 600 nm", "petal 0 at 800 nm"],
     "pl19_800_psf_zoo_piston_b", 800, 8.6)
lim = float(np.nanmax(np.abs(opd[[names.index(n) for n in LABEL if n != "flat wavefront"]])))
draw(["typical: p", "typical: twin Tp", "critical: p", "critical: twin Tp"],
     "pl19_800_psf_zoo_twins_a", lim, 8.6)
draw(["point-symmetric: p", "point-symmetric: -p (= its twin)"], "pl19_800_psf_zoo_twins_b", lim, 5.0)
for n in names:
    i = names.index(n)
    print(f"{n:36s} coupled {S[i].sum():.3f}  Strehl {SR[i]:.2f}  brightest core {int(np.argmax(S[i]))}")
