"""Is a pupil configuration distinguishable from its twin? (piston2.ipynb)

For N_BASE random petal-piston configurations p (uniform in +-SPAN_NM, so
that their twins T p stay within +-2 SPAN_NM), simulates the lantern signals of

    p              the configuration,
    T p            its twin (ml_reconstruct.petals: rotated by 180 degrees and
                   negated; same focal-plane intensity),
    p + D          a control at the same distance |D| = |T p - p| in a random
                   direction that, like the twin, leaves piston 2 unchanged,

as channel powers at each wavelength of WAVELENGTHS_NM (simulate_spectra), and
as the camera signals of Part A at 800 nm (simulate_core_images). One file per
wavelength in data/ml/piston2/ (restartable: existing files are skipped).

--family even does the same for N_EVEN point-symmetric configurations
(opposite petals equal, so piston 2 = 0 like its reference partner): their
twin is their sign flip, T p = -p, inside the training range.

--family lines simulates instead the signals along lines t v through the flat
wavefront (t in +-1, the largest piston +-lambda/2 = 400 nm at t = 1), for N_LINES
directions v of each kind: point-symmetric (T v = -v), twin-invariant
(T v = v) and random; at the 800 nm camera and at LINE_WAVELENGTHS_NM. A
fold s(t v) = s(-t v) makes the sign of t invisible.

--family examples simulates the 800 nm camera signals of the configurations
shown in piston2.ipynb's focal-plane examples (EXAMPLES: rows of
single_800nm_50k.npz, their twins and sign flips), in one batch with the
dataset's first N_CALIB configurations so that the core-centre calibration
matches the dataset's.

    python scripts/piston2_twins.py [--family {random,even,lines,examples}]
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
from ml_reconstruct import petals                                          # noqa: E402
from ml_reconstruct.presets import settings                              # noqa: E402
from ml_reconstruct.simulate import bp, simulate_core_images, simulate_spectra   # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "piston2")
# the single dataset's range +-lambda/2 (400 nm): bases in half of it, so that
# their twins stay inside it; even configurations and lines span all of it
RANGE_NM = settings("pl19_800")["single_span_nm"]
N_BASE, SPAN_NM, SEED = 400, RANGE_NM / 2, 11
N_EVEN, EVEN_SPAN_NM = 200, RANGE_NM
N_LINES, N_T, LINE_WAVELENGTHS_NM = 20, 25, (700.0, 800.0, 900.0)
EXAMPLES = {"typical": 45938, "critical": 43822}      # rows of single_800nm_50k.npz
N_CALIB = 2000
WAVELENGTHS_NM = np.arange(700.0, 901.0, 10.0)


def configurations(T, opposite, family="random"):
    """(3 n, 5) pistons: the n bases, their twins and the controls."""
    rng = np.random.default_rng(SEED)
    if family == "even":                    # p_k = p_opp(k); petal 2 pairs with the reference
        P = rng.uniform(-EVEN_SPAN_NM, EVEN_SPAN_NM, (N_EVEN, 5))
        for k in range(5):
            P[:, k] = 0.0 if opposite[k] == 5 else P[:, min(k, opposite[k])]
    else:
        P = rng.uniform(-SPAN_NM, SPAN_NM, (N_BASE, 5))
    PT = P @ T.T
    dist = np.linalg.norm(PT - P, axis=1, keepdims=True)
    D = rng.normal(size=P.shape)
    D[:, 2] = 0.0                                           # piston 2 fixed, like the twin
    D *= dist / np.linalg.norm(D, axis=1, keepdims=True)
    return np.concatenate([P, PT, P + D])


def line_configurations(T):
    """(3 N_LINES N_T, 5) pistons along the lines, and their (kind, line, t) labels."""
    rng = np.random.default_rng(SEED + 1)
    V = {"point-symmetric": rng.normal(size=(N_LINES, 5)), "twin-invariant": rng.normal(size=(N_LINES, 5)),
         "random": rng.normal(size=(N_LINES, 5))}
    I = np.eye(5)
    V["point-symmetric"] = V["point-symmetric"] @ ((I - T) / 2).T      # project on T v = -v
    V["twin-invariant"] = V["twin-invariant"] @ ((I + T) / 2).T        # project on T v = v
    t = np.linspace(-1, 1, N_T)
    C, labels = [], []
    for kind, vs in V.items():
        for j, v in enumerate(vs):
            v = v / np.abs(v).max() * SPAN_NM * 2                         # largest piston RANGE_NM at t = 1
            C.append(t[:, None] * v[None, :])
            labels += [(kind, j, tt) for tt in t]
    return np.concatenate(C), labels


def main_lines(T, ifunc, base):
    C, labels = line_configurations(T)
    kinds = np.array([k for k, _, _ in labels]); line = np.array([j for _, j, _ in labels])
    t = np.array([tt for _, _, tt in labels])
    path = os.path.join(OUT, "camera_lines_800nm.npz")
    if not os.path.exists(path):
        ds = simulate_core_images(bp.get_simulation_parameters(3, 0.8), ifunc, C, len(C))
        np.savez(path, signals=ds.signals, coeff=C, kind=kinds, line=line, t=t, T=T)
        print("lines: camera 800 nm done", flush=True)
    for wl in LINE_WAVELENGTHS_NM:
        path = os.path.join(OUT, f"channels_lines_{wl:05.1f}.npz")
        if not os.path.exists(path):
            ds = simulate_spectra(base, ifunc, C, len(C), [wl], verbose=False)
            np.savez(path, signals=ds.signals[:, 0], coeff=C, kind=kinds, line=line, t=t, T=T,
                     wavelength_nm=wl)
            print(f"lines: {wl:.0f} nm done", flush=True)


def example_configurations(T):
    """{name: pistons} of the examples: each row p, its twin T p and -p."""
    import ml_reconstruct as ml
    coeff = ml.LanternDataset.load(os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")).coeff
    out = {}
    for name, row in EXAMPLES.items():
        p = coeff[row]
        out.update({f"{name}: p": p, f"{name}: Tp": T @ p, f"{name}: -p": -p})
    return out, coeff[:N_CALIB]


def main_examples(T, ifunc):
    path = os.path.join(OUT, "camera_examples_800nm.npz")
    if os.path.exists(path):
        return
    ex, calib = example_configurations(T)
    C = np.concatenate([np.array(list(ex.values())), calib])
    ds = simulate_core_images(bp.get_simulation_parameters(3, 0.8), ifunc, C, len(C))
    np.savez(path, names=list(ex), coeff=C[:len(ex)], signals=ds.signals[:len(ex)],
             calib_signals=ds.signals[len(ex):], rows=[EXAMPLES[n.split(":")[0]] for n in ex])
    print("examples: camera 800 nm done", flush=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["random", "even", "lines", "examples"], default="random")
    family = ap.parse_args().family
    tag = "" if family == "random" else "_even"
    os.makedirs(OUT, exist_ok=True)
    base = bp.get_simulation_parameters(nrings=3)
    ifunc = bp.load_ifunc(base["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    g = petals.petal_geometry(ifunc.mask_inf_func, np.asarray(ifunc.influence_function))
    T = petals.twin_matrix(g["opposite"], g["ref"])
    if family == "lines":
        return main_lines(T, ifunc, base)
    if family == "examples":
        return main_examples(T, ifunc)
    C = configurations(T, g["opposite"], family)
    n_base = len(C) // 3
    path = os.path.join(OUT, f"camera{tag}_800nm.npz")
    if not os.path.exists(path):
        t = time.time()
        ds = simulate_core_images(bp.get_simulation_parameters(3, 0.8), ifunc, C, len(C))
        np.savez(path, signals=ds.signals, coeff=C, T=T, n_base=n_base)
        print(f"camera 800 nm done in {time.time() - t:.0f} s", flush=True)
    for wl in WAVELENGTHS_NM:
        path = os.path.join(OUT, f"channels{tag}_{wl:05.1f}.npz")
        if os.path.exists(path):
            continue
        t = time.time()
        ds = simulate_spectra(base, ifunc, C, len(C), [wl], verbose=False)
        np.savez(path, signals=ds.signals[:, 0], coeff=C, T=T, n_base=n_base, wavelength_nm=wl)
        print(f"{wl:.0f} nm done in {time.time() - t:.0f} s", flush=True)


if __name__ == "__main__":
    main()
