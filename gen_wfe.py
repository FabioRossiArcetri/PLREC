#!/usr/bin/env python
"""Single-wavelength datasets with a wavefront error besides the pistons.

The same 10000 piston configurations as the lantern's single-wavelength
dataset (rng=0, +-lambda/2), each with its own f^-2 phase screen
(ml_reconstruct.wavefront.psd_screens: global and petal pistons removed,
exact rms, highest frequency F_MAX cycles per pupil, which reproduces the
ANDES NCPA split of ~11.9 nm low-order / ~27.5 nm high-order at 30 nm rms).
The cores are measured at the clean dataset's pixel windows.

    CBEAM_LANTERN=pl19_1200 python gen_wfe.py piece <rms_nm> <chunk>   # one piece
    CBEAM_LANTERN=pl19_1200 python gen_wfe.py assemble <rms_nm>        # -> <data_dir>/single_wfe<rms>.npz

WFE_PSD=ao instead uses an SCAO residual (ml_reconstruct.wavefront.ao_residual_screens:
corrected modes below the M4 control frequency, 79/2 cycles per pupil, the uncorrected
fitting error beyond it) and writes <data_dir>/single_ao<rms>.npz (parts in ao_parts/).
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")

import ml_reconstruct as ml                                                     # noqa: E402
from ml_reconstruct.presets import settings                                     # noqa: E402
from ml_reconstruct.simulate import bp, simulate_core_images, dataset_centers   # noqa: E402
from ml_reconstruct.wavefront import psd_screens, ao_residual_screens           # noqa: E402

S = settings()
DATA = os.path.join(ROOT, S["data_dir"])
KIND = "ao" if os.environ.get("WFE_PSD") == "ao" else "wfe"   # the screen model and file prefix
TEST = os.environ.get("WFE_TEST") == "1"   # an independent test set: new pistons (rng=1) and screens
SUFFIX = "_test" if TEST else ""
PARTS = os.path.join(DATA, f"{KIND}{SUFFIX}_parts")
N, CHUNKS = (2000, 1) if TEST else (10000, 5)
F_MAX = 10.0                         # cycles per pupil diameter (see the docstring)


def out_path(rms):
    return os.path.join(DATA, f"single_{KIND}{rms:g}{SUFFIX}.npz")


def piece(rms, c):
    os.makedirs(PARTS, exist_ok=True)
    out = os.path.join(PARTS, f"{KIND}{rms:g}_c{c}.npz")
    if os.path.exists(out):
        print("already present:", out)
        return
    t = time.time()
    clean = ml.LanternDataset.load(os.path.join(ROOT, S["single"]))
    rows = slice(c * N // CHUNKS, (c + 1) * N // CHUNKS)
    p = bp.get_simulation_parameters(lantern=S["lantern"])
    ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    mask = np.asarray(ifunc.mask_inf_func) > 0
    pistons_if = np.asarray(ifunc.influence_function)[:5]
    n = rows.stop - rows.start
    seed = int(rms * 1000) + c + (500 if TEST else 0)   # one independent stream per (rms, chunk)
    coeff = (bp.create_random_aberration_configs(N, 5, -S["single_span_nm"], S["single_span_nm"], rng=1)[0][rows]
             if TEST else clean.coeff[rows])
    if KIND == "ao":
        opd = ao_residual_screens(n, mask, rms, rng=seed + 7, remove=pistons_if)
    else:
        opd = psd_screens(n, mask, rms, rng=seed, remove=pistons_if, f_max=F_MAX)
    ds = simulate_core_images(p, ifunc, coeff, 0, prop_chunk_size=250,
                              centers_by_port=dataset_centers(clean), extra_opd=opd,
                              field_chunk_size=8)   # bounded memory, many pieces in parallel
    tmp = out[:-4] + ".tmp.npz"
    np.savez(tmp, signals=ds.signals, coeff=coeff, seed=seed)
    os.replace(tmp, out)
    print(f"saved {out} {ds.signals.shape} ({time.time() - t:.0f} s)")


def assemble(rms):
    clean = ml.LanternDataset.load(os.path.join(ROOT, S["single"]))
    sig, coeff = [], []
    for c in range(CHUNKS):
        f = np.load(os.path.join(PARTS, f"{KIND}{rms:g}_c{c}.npz"))
        sig.append(f["signals"]); coeff.append(f["coeff"])
    coeff = np.concatenate(coeff)
    if not TEST:
        assert np.allclose(coeff, clean.coeff)
    meta = dict(clean.meta)
    meta.update(wavefront_error=(
        f"SCAO residual, {rms:g} nm rms: corrected band f < 39.5 cycles/D (41% of the variance, ~f^-5/3), "
        f"fitting error beyond (von Karman f^-11/3); petal pistons removed (wavefront.ao_residual_screens)"
        if KIND == "ao" else
        f"f^-2 PSD phase screens, {rms:g} nm rms, f_max {F_MAX:g} cycles/D, "
        f"global and petal pistons removed (ml_reconstruct.wavefront.psd_screens)"))
    ds = ml.LanternDataset(signals=np.concatenate(sig), coeff=coeff, n_train=0 if TEST else clean.n_train,
                           core_xy=clean.core_xy, wavelengths_nm=clean.wavelengths_nm, meta=meta)
    ds.save(out_path(rms))
    print("saved", out_path(rms), ds.signals.shape)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "piece":
        piece(float(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "assemble":
        assemble(float(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
