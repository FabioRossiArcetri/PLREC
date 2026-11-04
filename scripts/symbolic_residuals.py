"""Robustness of the analytical reconstructor to aberrations it does not model
(symbolic.ipynb): 800 nm camera signals of random petal pistons (+-300 nm) plus
random low-order residuals (tip, tilt, focus, astigmatism: modes 5-9 of the
modal basis) of a given total RMS, measured with the core centres of
single_800nm_50k.npz so that they match the signals the forward model was
fitted to. One file per residual level in data/ml/symbolic/.

    python scripts/symbolic_residuals.py       (a few minutes per level)
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml                                                        # noqa: E402
from ml_reconstruct import psfzoo                                                  # noqa: E402
from ml_reconstruct.simulate import bp, dataset_centers, simulate_core_images      # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "symbolic")
LEVELS_NM = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0)
N, SPAN_NM, SEED = 2000, 300.0, 21


def main():
    os.makedirs(OUT, exist_ok=True)
    p = bp.get_simulation_parameters(3, 0.8)
    ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    centers = dataset_centers(ml.LanternDataset.load(os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")))
    rng = np.random.default_rng(SEED)
    pist = rng.uniform(-SPAN_NM, SPAN_NM, (N, 5))
    shape = rng.normal(size=(N, 5))                     # low-order shapes, rescaled per level
    for level in LEVELS_NM:
        path = os.path.join(OUT, f"residuals_lo{level:g}.npz")
        if os.path.exists(path):
            continue
        lo = np.array([s * level / psfzoo.pupil_rms(ifunc, np.r_[np.zeros(5), s]) if level else 0 * s for s in shape])
        C = np.c_[pist, lo]
        ds = simulate_core_images(p, ifunc, C, N, centers_by_port=centers)
        np.savez(path, signals=ds.signals, coeff=C, level_nm=level)
        print(f"low-order residual {level:g} nm RMS done", flush=True)


if __name__ == "__main__":
    main()
