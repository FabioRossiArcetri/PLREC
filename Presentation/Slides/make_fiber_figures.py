#!/usr/bin/env python
"""The two figures of the slide "The lantern: 19 ports at 1200 nm", from data:

  pl19_1200_fiber_spectra     power in each of the 19 cores vs wavelength (41 wavelengths,
                              1100-1300 nm) for the zero-piston field of the ramp, with the sum
  pl19_1200_fiber_throughput  the same power summed over the band vs the amplitude of piston mode 0

Data: data/ml/pl19_1200/mw_ramp_41.npz (power_spectra [wavelength, field, core], 81 amplitudes of
mode 0 from -9600 to 9600 nm). The brightest core is drawn in the accent colour, the other 18 thin
and grey, so the slide reads at a glance and the colours do not compete with the method palette.
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
import slidefig                                                         # noqa: E402

f = np.load(os.path.join(ROOT, "data", "ml", "pl19_1200", "mw_ramp_41.npz"), allow_pickle=True)
P, wl = f["power_spectra"], f["wavelengths"]                # (41, 81, 19), (41,)
amp = f["coeff_matrix"][:, 0]
field = int(np.argmin(np.abs(amp)))                         # zero piston
band = P.sum(axis=0)                                        # (81, 19): integrated over wavelength
bright = int(np.argmax(band.mean(axis=0)))
others = [c for c in range(P.shape[2]) if c != bright]
ACC, GREY = slidefig.CYCLE[0], "0.62"
plt.rcParams.update(slidefig.RC)
W, H = slidefig.W_HALF * slidefig.CM, 4.9 * slidefig.CM


def draw(ax, x, y):
    for c in others:
        ax.plot(x, y[:, c], color=GREY, lw=0.5, alpha=0.8, zorder=1)
    ax.plot(x, y[:, bright], color=ACC, lw=1.1, zorder=3)
    ax.grid(alpha=0.3)


# ---------------------------------------------------------------- spectra of one field
fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
draw(ax, wl, P[:, field, :])
tot = P[:, field, :].sum(axis=1)
ax.plot(wl, tot, "--", color="0.15", lw=0.8, zorder=2)
ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("power (a.u.)")
ax.set_ylim(0, tot.max() * 1.25)
ax.legend([plt.Line2D([], [], color=ACC, lw=1.1), plt.Line2D([], [], color=GREY, lw=0.5),
           plt.Line2D([], [], color="0.15", lw=0.8, ls="--")],
          [f"core {bright}", "other 18 cores", "sum of 19"],
          loc="upper center", ncol=3, frameon=False, handlelength=1.2, columnspacing=0.8)
slidefig.export(fig, "pl19_1200_fiber_spectra", native=True)

# ---------------------------------------------------------------- band-integrated vs amplitude
fig, ax = plt.subplots(figsize=(W, H), layout="constrained")
draw(ax, amp / 1000, band)
ax.set_xlabel(r"piston of mode 0 ($\mu$m)"); ax.set_ylabel("power summed over the band (a.u.)")
ax.set_ylim(0, band.max() * 1.08)
ax.legend([plt.Line2D([], [], color=ACC, lw=1.1), plt.Line2D([], [], color=GREY, lw=0.5)],
          [f"core {bright} (brightest)", "the other 18 cores"], loc="center",
          bbox_to_anchor=(0.5, 0.34), ncol=2, frameon=False)
slidefig.export(fig, "pl19_1200_fiber_throughput", native=True)

print("field", field, "amp", amp[field], "bright core", bright,
      "band range", band[:, bright].min().round(1), band[:, bright].max().round(1),
      "others max", band[:, others].max().round(1),
      "total/41 min-max", (P.sum(2).mean(0)).min().round(3), (P.sum(2).mean(0)).max().round(3),
      "n wl", len(wl))
