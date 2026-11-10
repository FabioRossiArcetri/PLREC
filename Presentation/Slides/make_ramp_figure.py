#!/usr/bin/env python
"""The figure of the slide "Spectral response to one petal piston" (19 ports, 1200 nm):

  pl19_1200_core0_ramp_map   left: the central core's power (fraction of the incident power) as a map
                             of petal-0 piston (x) and wavelength (y), all 41 wavelengths;
                             right: three rows of that map (1100, 1200, 1300 nm) as curves, over
                             the central +-3 um, so the three fringe periods can be compared.

Data: data/ml/pl19_1200/mw_ramp_41_dense.npz (gen_ramp_dense.py: 961 pistons in 20 nm steps over
+-9.6 um, no other aberration).
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
import slidefig                                                         # noqa: E402

f = np.load(os.path.join(ROOT, "data", "ml", "pl19_1200", "mw_ramp_41_dense.npz"), allow_pickle=True)
P, wl, amp = f["power_spectra"][:, :, 0], f["wavelengths"], f["coeff_matrix"][:, 0] / 1000.0
plt.rcParams.update(slidefig.RC)
fig, ax = plt.subplots(1, 2, figsize=(14.0 * slidefig.CM, 4.0 * slidefig.CM), layout="constrained",
                       gridspec_kw=dict(width_ratios=[1.3, 1]))
im = ax[0].pcolormesh(amp, wl, P, cmap="viridis", shading="nearest", rasterized=True)
fig.colorbar(im, ax=ax[0], label="core-0 power", pad=0.01)
ax[0].set_xlabel(r"petal-0 piston ($\mu$m)"); ax[0].set_ylabel("wavelength (nm)")
ax[0].set_title(r"all 41 wavelengths, $\pm8\lambda$")
SHOW = [(1100, slidefig.CYCLE[0]), (1200, slidefig.CYCLE[1]), (1300, slidefig.CYCLE[2])]
for w, c in SHOW:
    ax[0].plot([amp.min()], [w], ">", color=c, ms=4, mec="k", mew=0.3, clip_on=False, zorder=5)
    ax[0].plot([amp.max()], [w], "<", color=c, ms=4, mec="k", mew=0.3, clip_on=False, zorder=5)
    i = int(np.argmin(abs(wl - w)))
    ax[1].plot(amp, P[i], color=c, lw=0.9, label=f"{w:.0f} nm")
ax[0].axvspan(-3, 3, ymin=0, ymax=1, fill=False, ec="w", lw=0.6, ls=":")
ax[1].set_xlim(-3, 3); ax[1].set_ylim(0, P.max() * 1.3)
ax[1].set_xlabel(r"petal-0 piston ($\mu$m)"); ax[1].set_ylabel("core-0 power")
ax[1].set_title(r"three rows of the map, central $\pm3\,\mu$m")
ax[1].legend(loc="upper center", ncol=3, frameon=False, handlelength=1.2, columnspacing=1.0,
             bbox_to_anchor=(0.5, 1.0)); ax[1].grid(alpha=0.3)
slidefig.export(fig, "pl19_1200_core0_ramp_map", native=True)
print("power range", P.min().round(3), P.max().round(3), "step nm", round((amp[1] - amp[0]) * 1000, 1))
