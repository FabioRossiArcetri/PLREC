"""The lantern's channel powers at 800 nm for every configuration of the PSF zoo
(ml_reconstruct.psfzoo.zoo) -> data/ml/psf_zoo/channels_800nm.npz, read by
psf_zoo.ipynb.

    python scripts/psf_zoo.py         (about a minute: one lantern characterization load)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml                                   # noqa: E402
from ml_reconstruct import petals, psfzoo                     # noqa: E402
from ml_reconstruct.simulate import bp, simulate_spectra      # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "psf_zoo")
EXAMPLE_ROWS = {"typical": 45938, "critical": 43822}          # rows of single_800nm_50k.npz (piston2.ipynb)
WAVELENGTH_NM = 800.0


def build():
    base = bp.get_simulation_parameters(nrings=3)
    ifunc = bp.load_ifunc(base["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    g = petals.petal_geometry(ifunc.mask_inf_func, np.asarray(ifunc.influence_function))
    T = petals.twin_matrix(g["opposite"], g["ref"])
    coeff = ml.LanternDataset.load(os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")).coeff
    return base, ifunc, psfzoo.zoo(ifunc, T, {k: coeff[r] for k, r in EXAMPLE_ROWS.items()})


def main():
    os.makedirs(OUT, exist_ok=True)
    base, ifunc, groups = build()
    names = [n for g in groups.values() for n in g]
    group_of = [k for k, g in groups.items() for _ in g]
    C = np.array([c for g in groups.values() for c in g.values()])
    ds = simulate_spectra(base, ifunc, C, len(C), [WAVELENGTH_NM], verbose=False)
    np.savez(os.path.join(OUT, "channels_800nm.npz"), names=names, groups=group_of, coeff=C,
             signals=ds.signals[:, 0], wavelength_nm=WAVELENGTH_NM)
    print(f"{len(C)} configurations done", flush=True)


if __name__ == "__main__":
    main()
