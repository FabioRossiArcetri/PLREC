#!/usr/bin/env python
"""Signal collisions vs. piston range for a lantern preset (no reconstructor involved).

    python diag_collisions.py <lantern> [n_samples]     # -> figures/sim/<lantern>_collisions.npz

For each piston range +-f*lambda (f in FRACS), draw n random configurations of
the 5 petal pistons (uniform, rng=0), compute the normalised output-core powers
at the design wavelength, and for every configuration find the OTHER one with
the closest signals. A collision is a pair whose pistons differ by more than
lambda/4 rms (per piston) MODULO LAMBDA: at one wavelength p and p +- lambda give
the same signals, so a neighbour across the wrap is the same configuration, not a
collision. (collision_plain: the same test without the wrap, as first reported.)
The signals alone would point to a wrong solution.
Also records how many collisions are twin-image pairs (closer to the twin of
the configuration than to the configuration itself). Needs the lantern's
characterization at its design wavelength in cbeam/examples/data.
"""
import contextlib
import io
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir, os.pardir)))
os.environ.setdefault("CBEAM_BACKEND", "numpy")
LANTERN = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
os.environ["CBEAM_LANTERN"] = LANTERN

from sklearn.neighbors import NearestNeighbors                     # noqa: E402
from ml_reconstruct.simulate import bp, CBEAM_EXAMPLES              # noqa: E402
from ml_reconstruct.petals import petal_geometry, twin_matrix       # noqa: E402
from ml_reconstruct.unwrap import wrap                              # noqa: E402

FRACS = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 5 / 12, 1 / 2]
quiet = contextlib.redirect_stdout(io.StringIO())
p = bp.get_simulation_parameters(lantern=LANTERN)
wl = p["wavelength_nm"]; nports = p["n_output_positions"]
ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
with quiet, contextlib.chdir(CBEAM_EXAMPLES):
    prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
    pipe = bp.BatchPropagationPipeline(prop12, p, ifunc)
g = petal_geometry(np.asarray(ifunc.mask_inf_func) > 0, np.asarray(ifunc.influence_function), 5)
T = twin_matrix(g["opposite"], g["ref"])


def signals(C):
    with quiet:
        u0 = pipe.generate_batch_modal_coefficients(C, use_gpu=False)
        uf, _, _ = pipe.propagate_batch(u0, chunk_size=500)
    a = np.stack([prop12.to_channel_basis(np.asarray(uf)[k]) for k in range(len(uf))])[:, :nports]
    P = np.abs(a) ** 2
    return P / P.sum(1, keepdims=True)


rms = lambda D: np.sqrt(np.mean(D ** 2, axis=-1))                  # noqa: E731
out = dict(fracs=np.array(FRACS), wavelength_nm=wl, n=N, collision=[], twin_share=[], collision_plain=[])
for f in FRACS:
    C = np.random.default_rng(0).uniform(-f * wl, f * wl, (N, 5))
    S = signals(C)
    _, j = NearestNeighbors(n_neighbors=2).fit(S).kneighbors(S)
    j = j[:, 1]                                                      # the closest OTHER configuration
    d, dt = rms(wrap(C[j] - C, wl)), rms(wrap(C[j] - C @ T.T, wl))  # piston distances modulo lambda
    coll = d > wl / 4
    out["collision"].append(coll.mean())
    out["collision_plain"].append((rms(C[j] - C) > wl / 4).mean())
    out["twin_share"].append((dt[coll] < d[coll]).mean() if coll.any() else 0.0)
    print(f"{LANTERN} +-{f:.3f} lambda: collisions {coll.mean():6.1%} (without the wrap "
          f"{out['collision_plain'][-1]:6.1%}), of which twin pairs {out['twin_share'][-1]:5.1%}", flush=True)
np.savez(os.path.join(HERE, "figures", "sim", f"{LANTERN}_collisions.npz"), **out)
