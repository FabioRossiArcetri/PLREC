#!/usr/bin/env python
"""A densely sampled ramp of petal piston 0 through the dispersed lantern, for the slide figure:
the power of every core at each of the 41 wavelengths, as a fraction of the incident power, for
N_STEPS pistons from -SPAN to +SPAN (no other aberration). Same format as mw_ramp_41.npz
(ml_reconstruct.simulate.ramp_spectra), which samples the same range in 240 nm steps -- too
coarse to show the fringes (about 5 samples per fringe).

    CBEAM_LANTERN=pl19_1200 python gen_ramp_dense.py piece <w>     # one wavelength, w = 0..40
    CBEAM_LANTERN=pl19_1200 python gen_ramp_dense.py assemble      # -> data/ml/<lantern>/mw_ramp_41_dense.npz
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
from ml_reconstruct.presets import settings                                   # noqa: E402
from ml_reconstruct.simulate import simulate_spectra                         # noqa: E402
from batch_pipeline import (get_simulation_parameters, load_ifunc,            # noqa: E402
                            center_ifunc_mask, create_ramp_aberration_configs)

S = settings()
BASE = get_simulation_parameters(lantern=S["lantern"])
WL = np.linspace(*BASE["band_nm"], 41)
SPAN, N_STEPS = 9600.0, 961                      # 20 nm steps over +-8 lambda at 1200 nm
OUT = os.path.join(HERE, os.path.dirname(S["ramp"]), "mw_ramp_41_dense.npz")
PARTS = os.path.join(HERE, os.path.dirname(S["ramp"]), "ramp_dense_parts")


def piece(w):
    coeff, _ = create_ramp_aberration_configs([0], N_STEPS, -SPAN, SPAN)
    f = load_ifunc(BASE["ifunc_file"]); center_ifunc_mask(f)
    ds = simulate_spectra(BASE, f, coeff, 0, [WL[w]], prop_chunk_size=250, verbose=False)
    os.makedirs(PARTS, exist_ok=True)
    np.save(os.path.join(PARTS, f"w{w:02d}.npy"), ds.signals[:, 0, :])     # (N_STEPS, n_cores)
    print("saved", w, WL[w], ds.signals.shape, flush=True)


def assemble():
    coeff, labels = create_ramp_aberration_configs([0], N_STEPS, -SPAN, SPAN)
    P = np.stack([np.load(os.path.join(PARTS, f"w{w:02d}.npy")) for w in range(len(WL))])
    np.savez(OUT, power_spectra=P, wavelengths=WL, coeff_matrix=coeff, labels=labels)
    print("saved", OUT, P.shape)


if __name__ == "__main__":
    if sys.argv[1] == "piece":
        piece(int(sys.argv[2]))
    else:
        assemble()
