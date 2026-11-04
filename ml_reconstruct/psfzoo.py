"""A zoo of PSFs at the lantern input (psf_zoo.ipynb): which configurations of
the DM modal basis, and how to draw their pupil OPD, focal-plane field and
lantern output.

The modal basis (ANDES_400pix_all_modes.fits) has 10 modes: 0-4 move one petal
each (unit piston: the coefficient is the petal's piston in nm; petal 6 is the
reference), 5-9 are smooth unit-RMS modes (the coefficient is in nm RMS):
tip, tilt, focus, astigmatism at 45 degrees and astigmatism at 0 degrees.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

N_MODES = 10
LOW_ORDER = {"tip": 5, "tilt": 6, "focus": 7, "astig 45": 8, "astig 0": 9}
# Residual levels of the SCAO-like cases: total RMS [nm] over the pupil of the
# residual in these 10 modes. The ANDES SCAO E2E simulations give a Strehl ratio
# of 77% at 1614 nm (I = 8, median seeing; Selmi et al., AO4ELT8, Table 1), i.e.
# about 131 nm RMS of total residual by the Marechal approximation -- over some
# 4000 corrected modes, so the share in 10 low-order modes is smaller. No
# per-mode breakdown is published there; 25 and 50 nm are assumptions (typical
# petal residuals quoted for good phasing), 131 nm the pessimistic bound of the
# whole budget in these modes.
RESIDUAL_LEVELS_NM = (25.0, 50.0, 131.0)
N_RESIDUAL_DRAWS = 3


def mode_vector(**coeffs):
    """10-vector of mode coefficients from keyword pistons p0..p4 / low-order names."""
    c = np.zeros(N_MODES)
    for k, v in coeffs.items():
        c[int(k[1:]) if k.startswith("p") and k[1:].isdigit() else LOW_ORDER[k.replace("_", " ")]] = v
    return c


def pupil_opd(ifunc, C):
    """(n, ny, nx) pupil OPD [nm] of the mode coefficients C (n, 10), NaN outside the pupil."""
    mask = np.asarray(ifunc.mask_inf_func) > 0
    M = np.asarray(ifunc.influence_function)
    out = np.full((len(C),) + mask.shape, np.nan)
    out[:, mask] = np.asarray(C, float) @ M
    return out


def pupil_rms(ifunc, c):
    """RMS [nm] over the pupil of the OPD of c, global piston removed."""
    opd = np.asarray(c, float) @ np.asarray(ifunc.influence_function)
    return float(np.std(opd))


def zoo(ifunc, T, coeff_rows, rng=0):
    """{group: {name: 10 mode coefficients [nm]}} of the zoo.

    T : the twin matrix of the 5 pistons (ml_reconstruct.petals.twin_matrix).
    coeff_rows : {name: 5 pistons} of the dataset configurations of the twin
        group (the "typical" and "critical" examples of piston2.ipynb).
    """
    rng = np.random.default_rng(rng)
    pad = lambda p: np.concatenate([np.asarray(p, float), np.zeros(N_MODES - 5)])
    groups = {"flat": {"flat wavefront": np.zeros(N_MODES)}}
    groups["single piston"] = {f"petal 0 at {a:g} nm": mode_vector(p0=a)
                               for a in (50, 100, 200, 300, 400, 600, 800)}
    twins = {}
    for name, p in coeff_rows.items():
        twins.update({f"{name}: p": pad(p), f"{name}: twin Tp": pad(T @ p)})
    sym = np.array([250.0, -250.0, 0.0, 250.0, -250.0])          # opposite petals equal: T p = -p
    twins.update({"point-symmetric: p": pad(sym), "point-symmetric: -p (= its twin)": pad(-sym)})
    groups["twin pairs"] = twins
    lo = {f"{m} {a:+g} nm": mode_vector(**{m.replace(" ", "_"): a}) for m in LOW_ORDER for a in (100,)}
    lo.update({"focus -100 nm": mode_vector(focus=-100), "focus +300 nm": mode_vector(focus=300)})
    groups["low order"] = lo
    res = {}
    for level in RESIDUAL_LEVELS_NM:
        for j in range(N_RESIDUAL_DRAWS):
            c = rng.normal(size=N_MODES)
            res[f"{level:g} nm RMS residual, draw {j + 1}"] = c * level / pupil_rms(ifunc, c)
    groups["SCAO-like residuals"] = res
    return groups


def strehl(E, E_flat):
    """Peak focal-plane intensity relative to the flat wavefront (same total power)."""
    return np.abs(E).reshape(len(E), -1).max(1) ** 2 / np.abs(E_flat).max() ** 2


def plot_zoo(names, opd, E, ext, signals, E_flat, r_clad_um=9.0, opd_lim=None):
    """One row per configuration: pupil OPD, focal-plane intensity (log, relative
    to the flat peak), focal-plane field (phase as hue, intensity as
    brightness), and the lantern's 19 channel powers (fraction of the
    incident power; title: total coupled and Strehl). The phase is drawn with
    the intensity as brightness (piston2.complex_image), so no masking is
    needed where the field is weak."""
    from .piston2 import complex_image, phase_colorbar
    n = len(names)
    fig, ax = plt.subplots(n, 4, figsize=(15, 3.0 * n), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]),
                           squeeze=False)
    I0 = np.abs(E_flat).max() ** 2
    lim = opd_lim or max(np.nanmax(np.abs(opd)), 1.0)
    sr = strehl(E, E_flat)
    for i, name in enumerate(names):
        I = np.abs(E[i]) ** 2
        im = ax[i, 0].imshow(opd[i], cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax[i, 0].set_title(name, fontsize=9); ax[i, 0].axis("off")
        plt.colorbar(im, ax=ax[i, 0], fraction=0.046, label="OPD (nm)")
        ax[i, 1].imshow(np.log10(I / I0 + 1e-6), extent=ext, origin="lower", cmap="inferno", vmin=-4, vmax=0)
        ax[i, 2].imshow(complex_image(E[i]), extent=ext, origin="lower")
        phase_colorbar(ax[i, 2])
        for a in ax[i, 1:3]:
            a.add_patch(plt.Circle((0, 0), r_clad_um, fill=False, ec="c", lw=1, ls="--"))
        ax[i, 3].bar(np.arange(len(signals[i])), signals[i], color="tab:blue")
        ax[i, 3].set_xticks(range(len(signals[i])))
        ax[i, 3].set_title(f"coupled {signals[i].sum():.1%} of the light, Strehl {sr[i]:.2f}", fontsize=9)
        ax[i, 3].grid(alpha=0.3, axis="y")
    ax[0, 1].set_title("intensity, log10 rel. to flat peak\n(dashed: lantern input)", fontsize=9)
    ax[0, 2].set_title("focal-plane field: hue = phase,\nbrightness = intensity (4 decades)", fontsize=9)
    ax[-1, 1].set_xlabel("um"); ax[-1, 2].set_xlabel("um"); ax[-1, 3].set_xlabel("output core")
    plt.tight_layout(); plt.show()


def pair_table(names, E, signals, pairs, scale):
    """For pairs (a, b) of configuration names: the largest focal-plane
    intensity difference (fraction of a's peak), the median |phase(a) +
    phase(b)| (0 if b is the complex conjugate of a), and the distance of the
    lantern outputs in units of ``scale`` (e.g. the median distance between two
    random configurations of the dataset)."""
    import pandas as pd
    i = {n: k for k, n in enumerate(names)}
    rows = []
    for a, b in pairs:
        Ia, Ib = np.abs(E[i[a]]) ** 2, np.abs(E[i[b]]) ** 2
        on = Ia > 1e-3 * Ia.max()
        rows.append(dict(a=a, b=b, intensity_diff=np.abs(Ia - Ib).max() / Ia.max(),
                         conj_phase_rad=np.median(np.abs(np.angle(E[i[a]] * E[i[b]])[on])),
                         output_distance=np.linalg.norm(signals[i[a]] - signals[i[b]]) / scale))
    return pd.DataFrame(rows)


def random_pair_scale(S, n=2000, rng=0):
    """Median distance between the outputs of two random configurations of S (n_samples, n_cores)."""
    rng = np.random.default_rng(rng)
    i, j = rng.integers(0, len(S), (2, n))
    return float(np.median(np.linalg.norm(S[i] - S[j], axis=1)))
