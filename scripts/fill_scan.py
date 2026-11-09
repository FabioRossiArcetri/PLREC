"""PSF size on the lantern face vs coupling and single-wavelength sensing.

The PSF fill (Airy first-null radius / cladding radius, batch_pipeline config:
psf_fill_factor) is a property of the feeding optics, not of the fibre: no new
characterization is needed. For each fill this simulates N random petal-piston
configurations in +-lambda/2 at the design wavelength, fits one cubic ridge
regressor per piston phasor (sin, cos)(2 pi p / lambda) -- the per-wavelength
model of the dispersed reconstructor, free of the +-lambda/2 wrap -- and
reports the wrapped RMSE on held-out samples, with the flat-wavefront coupling.

    python scripts/fill_scan.py <lantern> [fill ...]      (default fills 0.20 .. 0.60)
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
LANTERN = sys.argv[1]
os.environ["CBEAM_LANTERN"] = LANTERN
os.environ.setdefault("CBEAM_BACKEND", "numpy")
from ml_reconstruct.simulate import bp, CBEAM_EXAMPLES, simulate_core_images   # noqa: E402
from ml_reconstruct.sparse import fit_dense_poly_ridge, predict_dense_poly_ridge  # noqa: E402
from ml_reconstruct.unwrap import phasor_targets, phasor_decode, wrap          # noqa: E402

FILLS = [float(f) for f in sys.argv[2:]] or [0.20, 0.30, 0.40, 0.50, 0.60]
N, N_TR = int(os.environ.get("FILL_N", 6000)), int(os.environ.get("FILL_NTR", 5000))

quiet = contextlib.redirect_stdout(io.StringIO())
p0 = bp.get_simulation_parameters(lantern=LANTERN)
wl = p0["wavelength_nm"]
ifunc = bp.load_ifunc(p0["ifunc_file"]); bp.center_ifunc_mask(ifunc)
with quiet, contextlib.chdir(CBEAM_EXAMPLES):
    prop12 = bp.build_and_characterize_lantern(p0, reuse_cache=True)
coeff, _ = bp.create_random_aberration_configs(N, 5, -wl / 2, wl / 2, rng=0)
flat = np.zeros((1, coeff.shape[1]))
print(f"{LANTERN} at {wl:.0f} nm, design fill {p0['psf_fill_factor']:.3f}; {N_TR} train / {N - N_TR} test, +-lambda/2")
print(f"{'fill':>6s} {'coupling':>9s} {'wrapped RMSE nm':>16s} {'% of lambda':>11s} {'per piston':>34s} {'s':>5s}")
for fill in FILLS:
    t0 = time.time()
    p = dict(p0)
    p["psf_fill_factor"] = fill
    p["pixel_scale_um"] = fill * p["rclad"] / (1.22 * p["pad_factor"])
    with quiet:
        pipe = bp.BatchPropagationPipeline(prop12, p, ifunc)
        u, eff = pipe.generate_batch_modal_coefficients(flat, use_gpu=False, return_coupling_efficiency=True)
        # power in the guided modes, as in the slides (make_figures.py psf_coupling)
        eff = np.sum(np.abs(np.asarray(u) * np.asarray(eff)[:, None]) ** 2)
        ds = simulate_core_images(p, ifunc, coeff, N_TR, prop12=prop12, prop_chunk_size=1000)
    X = np.asarray(ds.signals)
    Ytr = phasor_targets(coeff[:N_TR], wl)
    model = fit_dense_poly_ridge(X[:N_TR], Ytr, 3)
    est, _ = phasor_decode(predict_dense_poly_ridge(model, X[N_TR:]), wl)
    err = wrap(est - coeff[N_TR:], wl)
    per = np.sqrt(np.mean(err ** 2, axis=0))
    print(f"{fill:6.2f} {100 * float(np.ravel(eff)[0]):8.1f}% {per.mean():16.2f} {100 * per.mean() / wl:10.2f}% "
          f"{np.array2string(per, precision=1):>34s} {time.time() - t0:5.0f}", flush=True)
