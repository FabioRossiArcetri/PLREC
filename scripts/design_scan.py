"""Fibre-design variants of the 19-port 1200 nm lantern, scored at one wavelength.

Each variant is pl19_1200 with some parameters changed. It is registered as a
batch_pipeline preset at run time (own cache tag, mode bookkeeping inferred),
characterized at 1200 nm only (multi_wvl), and scored like fill_scan.py: N
random petal-piston configurations in +-lambda/2, one cubic ridge regressor per
piston phasor, wrapped RMSE (% of lambda) on held-out samples, and the
flat-wavefront coupling -- at the default PSF size (fill rule) and at fill 0.5.

    python scripts/design_scan.py <variant>        (variants: see VARIANTS)
"""
import contextlib
import io
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
os.environ["CBEAM_LANTERN"] = "pl19_1200"
from ml_reconstruct.simulate import bp, CBEAM_EXAMPLES, simulate_core_images   # noqa: E402
from ml_reconstruct.sparse import fit_dense_poly_ridge, predict_dense_poly_ridge  # noqa: E402
from ml_reconstruct.unwrap import phasor_targets, phasor_decode, wrap           # noqa: E402
from multi_wvl import build_and_characterize_lantern_at_wavelength              # noqa: E402

BASE = dict(bp.LANTERNS["pl19_1200"])
VARIANTS = {
    "base": {},
    "r9":   dict(rclad=9.0, rjack=27.0),                                  # V 8.0
    "t12":  dict(taper_factor=12.0, rcore=0.22 * 15 / 12),                # same core at the SM end
    "t20":  dict(taper_factor=20.0, rcore=0.22 * 15 / 20),
    "g800": dict(nclad=1.444, dn_core=8.8e-3, dn_jack=-5.5e-3, rclad=13.5, rjack=40.5,
                 rcore=1.8 * 1.5 / 12, taper_factor=12.0),                # the 800 nm design x1.5
    "c88":  dict(dn_core=8.8e-3),
}
NAME = sys.argv[1]
N, N_TR = int(os.environ.get("DESIGN_N", 6000)), int(os.environ.get("DESIGN_NTR", 5000))
lantern = f"dz_{NAME}"
bp.LANTERNS[lantern] = dict(BASE, cache_tag=lantern, auto_bookkeeping=True, **VARIANTS[NAME])

quiet = contextlib.redirect_stdout(io.StringIO())
t0 = time.time()
base = bp.get_simulation_parameters(lantern=lantern)
wl = base["wavelength_nm"]
log = io.StringIO()
with contextlib.redirect_stdout(log), contextlib.chdir(CBEAM_EXAMPLES):
    p, prop12 = build_and_characterize_lantern_at_wavelength(base, wl)
book = [ln for ln in log.getvalue().splitlines() if "skip" in ln.lower() or "degenerate" in ln.lower()]
print(f"{NAME}: V {base['V_mmf']:.2f}, characterized in {time.time() - t0:.0f} s; " + " | ".join(book[-2:]), flush=True)

ifunc = bp.load_ifunc(base["ifunc_file"]); bp.center_ifunc_mask(ifunc)
coeff, _ = bp.create_random_aberration_configs(N, 5, -wl / 2, wl / 2, rng=0)
flat = np.zeros((1, coeff.shape[1]))
for fill in (p["psf_fill_factor"], 0.5):
    t = time.time()
    q = dict(p, psf_fill_factor=fill, pixel_scale_um=fill * p["rclad"] / (1.22 * p["pad_factor"]))
    with quiet:
        pipe = bp.BatchPropagationPipeline(prop12, q, ifunc)
        u, eff = pipe.generate_batch_modal_coefficients(flat, use_gpu=False, return_coupling_efficiency=True)
        coupling = np.sum(np.abs(np.asarray(u) * np.asarray(eff)[:, None]) ** 2)
        ds = simulate_core_images(q, ifunc, coeff, N_TR, prop12=prop12, prop_chunk_size=1000)
    X = np.asarray(ds.signals)
    model = fit_dense_poly_ridge(X[:N_TR], phasor_targets(coeff[:N_TR], wl), 3)
    est, _ = phasor_decode(predict_dense_poly_ridge(model, X[N_TR:]), wl)
    per = np.sqrt(np.mean(wrap(est - coeff[N_TR:], wl) ** 2, axis=0))
    print(f"RESULT {NAME:5s} fill {fill:.3f}: coupling {100 * coupling:5.1f}%, wrapped RMSE "
          f"{per.mean():6.2f} nm = {100 * per.mean() / wl:.2f}% of lambda  {np.round(per, 1)}  "
          f"({time.time() - t:.0f} s)", flush=True)
