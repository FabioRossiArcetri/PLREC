#!/usr/bin/env python
"""Spectral dataset with a wavefront error besides the pistons, for the per-wavelength
models + unwrap.

The first N piston configurations of the spectral dataset (rng=0, +-8 lambda), each with
its own f^-2 phase screen of RMS nm rms (ml_reconstruct.wavefront.psd_screens, as
gen_wfe.py: global and petal pistons removed, highest frequency 10 cycles per pupil).
The screen is an OPD in nm, the same at every wavelength. Channel powers as a fraction
of the incident power (simulate_spectra), at the wavelengths the unwrap uses (every
second of the 41). Needs the lantern's characterization at those wavelengths.

    CBEAM_LANTERN=pl19_1200 python gen_wfe_spectral.py piece <rms> <w> <chunk>
    CBEAM_LANTERN=pl19_1200 python gen_wfe_spectral.py assemble <rms>
    (WFE_PSD=ao: SCAO residual screens, WFE_INBAND: their variance share below f_c, default 0.41)
        -> <data_dir>/spectral_wfe<rms>_21wl.npz   (N, 21, n_cores), first N_TRAIN rows train
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")

from ml_reconstruct.presets import settings                                   # noqa: E402
from ml_reconstruct.simulate import simulate_spectra, output_core_xy          # noqa: E402
from ml_reconstruct.wavefront import psd_screens, ao_residual_screens         # noqa: E402
from batch_pipeline import (get_simulation_parameters, load_ifunc,            # noqa: E402
                            center_ifunc_mask, create_random_aberration_configs)

S = settings()
BASE = get_simulation_parameters(lantern=S["lantern"])
DATA = os.path.join(ROOT, S["data_dir"])
KIND = "ao" if os.environ.get("WFE_PSD") == "ao" else "wfe"   # WFE_PSD=ao: SCAO residual screens
INBAND = float(os.environ.get("WFE_INBAND", "0.41"))   # SCAO residual: variance share below f_c (0.46: the ANDES pyramid loop)
TEST = os.environ.get("WFE_TEST") == "1"   # an independent test set: new pistons (rng=1) and screens
SUFFIX = "_test" if TEST else ""
PARTS = os.path.join(DATA, f"{KIND}{SUFFIX}_spectral_parts")
WL = np.linspace(*BASE["band_nm"], 41)[::2]          # the 21 wavelengths of the unwrap
N, N_TRAIN, CHUNKS = (2000, 0, 1) if TEST else (10000, 8000, 5)
N_SPECTRAL = 50000                                    # the spectral dataset's draw (rng=0)
F_MAX = 10.0


def _ifunc():
    f = load_ifunc(BASE["ifunc_file"]); center_ifunc_mask(f)
    return f


def piece(rms, w, c):
    os.makedirs(PARTS, exist_ok=True)
    out = os.path.join(PARTS, f"{KIND}{rms:g}_wl{w:02d}_c{c}.npz")
    if os.path.exists(out):
        print("already present:", out); return
    t = time.time()
    rows = slice(c * N // CHUNKS, (c + 1) * N // CHUNKS)
    span = S["spectral_span_nm"]
    coeff = (create_random_aberration_configs(N, 5, -span, span, rng=1)[0][rows] if TEST else
             create_random_aberration_configs(N_SPECTRAL, 5, -span, span, rng=0)[0][:N][rows])
    ifunc = _ifunc()
    mask = np.asarray(ifunc.mask_inf_func) > 0
    seed = int(rms * 1000) + 100 + c + (500 if TEST else 0)                                            # one stream per chunk:
    pist = np.asarray(ifunc.influence_function)[:5]                              # same at every wl
    opd = (ao_residual_screens(len(coeff), mask, rms, rng=seed + 7, remove=pist, inband_share=INBAND) if KIND == "ao"
           else psd_screens(len(coeff), mask, rms, rng=seed, remove=pist, f_max=F_MAX))
    ds = simulate_spectra(BASE, ifunc, coeff, 0, [WL[w]], prop_chunk_size=250, field_gen_chunk_size=8,
                          extra_opd=opd, verbose=False)
    tmp = out[:-4] + ".tmp.npz"
    np.savez(tmp, signals=ds.signals[:, 0, :], coeff=coeff, wavelength_nm=WL[w])
    os.replace(tmp, out)
    print(f"saved {out} ({time.time() - t:.0f} s)")


def assemble(rms):
    from ml_reconstruct.data import LanternDataset
    sig = np.stack([np.concatenate([np.load(os.path.join(PARTS, f"{KIND}{rms:g}_wl{w:02d}_c{c}.npz"))["signals"]
                                    for c in range(CHUNKS)]) for w in range(len(WL))], axis=1)
    coeff = np.concatenate([np.load(os.path.join(PARTS, f"{KIND}{rms:g}_wl00_c{c}.npz"))["coeff"] for c in range(CHUNKS)])
    ds = LanternDataset(signals=sig, coeff=coeff, n_train=0 if TEST else N_TRAIN, core_xy=output_core_xy(BASE),
                        wavelengths_nm=WL,
                        meta={"measurement": "channel power / incident power, per wavelength",
                              "lantern": S["lantern"],
                              "wavefront_error": (f"SCAO residual ({KIND}, in-band share {INBAND:g}), " if KIND == "ao" else "f^-2 screens, ") + f"{rms:g} nm rms, f_max {F_MAX:g} cycles/D, "
                                                 f"global and petal pistons removed; same OPD at every wavelength"})
    out = os.path.join(DATA, f"spectral_{KIND}{rms:g}_21wl{SUFFIX}.npz")
    ds.save(out)
    print("saved", out, sig.shape)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "piece":
        piece(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif cmd == "assemble":
        assemble(float(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
