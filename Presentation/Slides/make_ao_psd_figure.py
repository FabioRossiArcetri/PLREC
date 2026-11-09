#!/usr/bin/env python
"""The SCAO-residual model used for the wavefront error besides the pistons (appendix slide).

Left: the radial PSD of ml_reconstruct.wavefront.ao_residual_psd, scaled to 120 nm rms on the
pupil grid -- the corrected band below the M4 control frequency (79 actuators across the pupil:
f_c = 39.5 cycles per pupil) and the uncorrected fitting error beyond it; for comparison the
uncorrected von Karman spectrum and the earlier f^-2 model. Right: one 120 nm screen (petal
pistons removed) on the ELT pupil.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir, os.pardir)))
os.environ.setdefault("CBEAM_BACKEND", "numpy")
import slidefig                                                        # noqa: E402
from ml_reconstruct.simulate import bp                                 # noqa: E402
from ml_reconstruct.wavefront import ao_residual_psd, ao_residual_screens   # noqa: E402

N_ACT, RMS = 79, 120.0
FC = N_ACT / 2
p = bp.get_simulation_parameters(lantern="pl19_1200")
ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
mask = np.asarray(ifunc.mask_inf_func) > 0
screen = ao_residual_screens(1, mask, RMS, rng=3, remove=np.asarray(ifunc.influence_function)[:5])[0]

# radial PSDs on a fine radial grid, in arbitrary units scaled so the AO model integrates to RMS^2
f = np.logspace(-0.5, np.log10(200), 600)                 # cycles per pupil (grid Nyquist ~200)
p_in, p_out = ao_residual_psd(f, N_ACT)
share = 0.41
w_in = share / np.trapezoid(2 * np.pi * f * p_in, f)
w_out = (1 - share) / np.trapezoid(2 * np.pi * f * p_out, f)
psd_ao = RMS ** 2 * (w_in * p_in + w_out * p_out)
vk = (f ** 2 + (39.0 / 25.0) ** 2) ** (-11 / 6)
vk *= psd_ao[np.argmin(np.abs(f - 80))] / vk[np.argmin(np.abs(f - 80))]   # matched beyond f_c
f2 = f ** -2.0 * psd_ao[np.argmin(np.abs(f - 3))] / 3 ** -2.0

plt.rcParams.update(slidefig.RC)
fig, ax = plt.subplots(1, 2, figsize=(13.0 * slidefig.CM, 5.2 * slidefig.CM), layout="constrained",
                       gridspec_kw=dict(width_ratios=[1.6, 1]))
a = ax[0]
a.loglog(f, vk, color="0.6", ls="--", lw=0.8, label=r"turbulence, no AO ($f^{-11/3}$)")
a.loglog(f[f < FC], psd_ao[f < FC], color="C0", lw=1.4, label=r"corrected modes ($\sim f^{-5/3}$)")
a.loglog(f[f >= FC], psd_ao[f >= FC], color="C3", lw=1.4, label=r"fitting error ($f^{-11/3}$)")
a.loglog(f, f2, color="C2", ls=":", lw=0.9, label=r"earlier model ($f^{-2}$)")
a.axvline(FC, color="k", lw=0.6)
a.text(FC * 1.05, a.get_ylim()[1] if False else psd_ao.max() * 0.3, r" $f_c$ = 39.5" "\n" r" (79 act.)", fontsize=5.6)
a.axvspan(0.3, 2.4, color="C1", alpha=0.12, lw=0)
a.text(0.36, 3e-2, "within the\nlantern core\n($<$2.4 $\\lambda/D$)", fontsize=5.0, color="C1")
a.set_xlabel(r"spatial frequency (cycles per pupil = $\lambda/D$ in the focal plane)")
a.set_ylabel(r"PSD (nm$^2$ per (cycle/pupil)$^2$)")
a.set_xlim(f[0], f[-1]); a.grid(alpha=0.3, which="both")
a.legend(loc="upper right", fontsize=5.2)
b = ax[1]
img = np.full(mask.shape, np.nan); img[mask] = screen
im = b.imshow(img, cmap="RdBu_r", vmin=-3 * RMS, vmax=3 * RMS, origin="upper")
b.set_title(f"one screen, {RMS:.0f} nm rms", fontsize=6); b.axis("off")
fig.colorbar(im, ax=b, shrink=0.8, label="OPD (nm)")
slidefig.export(fig, "ao_residual_psd", native=True)
print("saved")
