#!/usr/bin/env python
"""Generate a lantern's datasets: the single-wavelength one and the chunked spectral one.

The lantern is CBEAM_LANTERN (a batch_pipeline.LANTERNS name; run_gen.sh sets
it), and its characterizations must exist in the cbeam checkout
(cbeam/characterize.py). Paths come from ml_reconstruct.presets:

    python gen_datasets.py single                        # <data_dir>/single*.npz (10k)
    python gen_datasets.py spectral-chunk <W> <C>        # one (wavelength, chunk) piece
    python gen_datasets.py spectral-assemble             # pieces -> <data_dir>/spectral_41wl.npz
    python gen_datasets.py peak-memory                   # a short piece, prints its peak RSS

The spectral dataset is 50000 samples x 41 wavelengths x n_cores, split into
41 wavelengths x 10 chunks of 5000 samples so a many-core host can run the 410
pieces in parallel (run_gen.sh). Every piece draws the same rng=0
coefficients, so the pieces align row-for-row at assembly.
"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from ml_reconstruct.presets import settings                                    # noqa: E402
from ml_reconstruct.simulate import make_single_dataset, simulate_spectra       # noqa: E402
from batch_pipeline import (get_simulation_parameters, load_ifunc,              # noqa: E402
                            center_ifunc_mask, create_random_aberration_configs)  # noqa: E402

S = settings()                                  # CBEAM_LANTERN
BASE = get_simulation_parameters(lantern=S["lantern"])
DATA = os.path.join(ROOT, S["data_dir"])
PARTS = os.path.join(DATA, "spectral_parts")
WL = np.linspace(*BASE["band_nm"], 41)          # 41 native wavelengths
N_SINGLE, N_SINGLE_TRAIN = 10000, 8000
N_TOTAL, N_TRAIN = 50000, 40000
N_MODES = 5                                     # the ANDES DM's 5 petal pistons
SPAN_NM = S["spectral_span_nm"]                 # spectral: +-8 lambda
CHUNKS = 10
# Memory per spectral piece: field generation holds FIELD_CHUNK padded FFT
# grids (1600 x 1600 complex) at a time, propagation PROP_CHUNK fields.
# (250 fields per propagation chunk: the solver keeps every step of a chunk, so this
#  roughly halves that part of the peak; results agree with 500 to the solver tolerance)
FIELD_CHUNK, PROP_CHUNK = 8, 250


def _ifunc():
    ifunc = load_ifunc(BASE["ifunc_file"])
    center_ifunc_mask(ifunc)
    return ifunc


def do_single():
    out = os.path.join(ROOT, S["single"])
    if os.path.exists(out):
        print("already present:", out)
        return
    t = time.time()
    ds = make_single_dataset(N_SINGLE, N_SINGLE_TRAIN, S["single_span_nm"], lantern=S["lantern"],
                             prop_chunk_size=1000)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ds.save(out[:-4] + ".tmp.npz")
    os.replace(out[:-4] + ".tmp.npz", out)
    print(f"saved {out}  {ds.signals.shape}  ({time.time()-t:.0f} s)")


def _piece(w, rows):
    coeff, _ = create_random_aberration_configs(N_TOTAL, N_MODES, -SPAN_NM, SPAN_NM, rng=0)
    coeff = coeff[rows]
    ds = simulate_spectra(BASE, _ifunc(), coeff, len(coeff), [WL[w]], prop_chunk_size=PROP_CHUNK,
                          field_gen_chunk_size=FIELD_CHUNK, verbose=False)
    return ds, coeff


def do_spectral_chunk(w, c):
    os.makedirs(PARTS, exist_ok=True)
    out = os.path.join(PARTS, f"wl{w:02d}_c{c:02d}.npz")
    if os.path.exists(out):
        print("already present:", out)
        return
    t = time.time()
    ds, coeff = _piece(w, slice(c * (N_TOTAL // CHUNKS), (c + 1) * (N_TOTAL // CHUNKS)))
    # written under a temporary name and renamed: a piece that exists is complete
    # (pieces made elsewhere can be copied in while a run is going on)
    tmp = out[:-4] + ".tmp.npz"
    np.savez(tmp, signals=ds.signals, coeff=coeff, wavelength_nm=WL[w])
    os.replace(tmp, out)
    print(f"saved {out}  {ds.signals.shape}  ({time.time()-t:.0f} s)")


def do_peak_memory():
    """Run a 1000-sample piece and report the peak resident memory, to size
    the number of parallel pieces (run_gen.sh)."""
    import resource
    t = time.time()
    _piece(len(WL) // 2, slice(0, 1000))
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20   # Linux: kB
    print(f"PEAK_GB {peak_gb:.2f}  ({time.time()-t:.0f} s for 1000 samples)")


def do_spectral_assemble():
    sigs, wls = [], []
    for w in range(len(WL)):
        col = []
        for c in range(CHUNKS):
            f = np.load(os.path.join(PARTS, f"wl{w:02d}_c{c:02d}.npz"))
            col.append(f["signals"][:, 0, :])            # (n, n_cores)
        sigs.append(np.concatenate(col, axis=0))         # (50000, n_cores)
        wls.append(float(np.load(os.path.join(PARTS, f"wl{w:02d}_c00.npz"))["wavelength_nm"]))
    sigs = np.stack(sigs, axis=1)                        # (50000, 41, n_cores)

    coeff_all, _ = create_random_aberration_configs(N_TOTAL, N_MODES, -SPAN_NM, SPAN_NM, rng=0)
    for w in range(len(WL)):                             # the pieces must align row for row
        for c in range(CHUNKS):
            rows = slice(c * (N_TOTAL // CHUNKS), (c + 1) * (N_TOTAL // CHUNKS))
            assert np.allclose(np.load(os.path.join(PARTS, f"wl{w:02d}_c{c:02d}.npz"))["coeff"],
                               coeff_all[rows]), (w, c)

    from ml_reconstruct.data import LanternDataset
    from ml_reconstruct.simulate import output_core_xy
    ds = LanternDataset(signals=sigs, coeff=coeff_all, n_train=N_TRAIN,
                        core_xy=output_core_xy(BASE), wavelengths_nm=np.asarray(wls),
                        meta={"measurement": "channel power / incident power, per wavelength",
                              "lantern": S["lantern"]})
    out = os.path.join(ROOT, S["spectral"])
    ds.save(out)
    print(f"saved {out}  signals {sigs.shape}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"lantern {S['lantern']}, data in {DATA}", flush=True)
    if cmd == "single":
        do_single()
    elif cmd == "spectral-chunk":
        do_spectral_chunk(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "spectral-assemble":
        do_spectral_assemble()
    elif cmd == "peak-memory":
        do_peak_memory()
    else:
        raise SystemExit(__doc__)
