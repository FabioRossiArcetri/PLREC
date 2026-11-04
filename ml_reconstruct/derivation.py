"""Step-by-step derivation of the single-wavelength closed form (derivation.ipynb):
sympy expressions of each step, symbolic checks, and the first-principles
computation of the constants from cbeam (no fitting).

Notation: petals j = 0..5 (5 = the reference R), piston p_j [nm], phase
theta_j = 2 pi p_j / lambda, phasor z_j = exp(i theta_j).
"""
from __future__ import annotations

import contextlib
from itertools import combinations

import numpy as np
import sympy as sp

N_PETALS = 6
PAIRS = list(combinations(range(N_PETALS), 2))


# --------------------------------------------------------------------------- #
# symbols
# --------------------------------------------------------------------------- #
def symbols(n=N_PETALS):
    """Petal phases theta_j, pistons p_j, the wavelength, and petal indicators Pi_j."""
    theta = sp.symbols(f"theta_0:{n}", real=True)
    p = sp.symbols(f"p_0:{n}", real=True)
    lam = sp.Symbol("lambda", positive=True)
    Pi = sp.symbols(f"Pi_0:{n}")
    return theta, p, lam, Pi


def generic():
    """Indexed objects for the generic (any number of petals) formulas."""
    j, k, m, c = sp.symbols("j k m c", integer=True)
    n = sp.Symbol("N", integer=True, positive=True)
    th = sp.IndexedBase("theta", real=True)
    return j, k, m, c, n, th


# --------------------------------------------------------------------------- #
# step 1: the pupil field
# --------------------------------------------------------------------------- #
def pupil_phase_identity(n=N_PETALS):
    """Check exp(i sum_j theta_j Pi_j) == sum_j Pi_j exp(i theta_j) at every
    point of the pupil, where exactly one indicator Pi_j is 1 (disjoint petals
    covering the pupil). Returns (lhs, rhs, [difference at each petal])."""
    theta, _, _, Pi = symbols(n)
    lhs = sp.exp(sp.I * sum(t * q for t, q in zip(theta, Pi)))
    rhs = sum(q * sp.exp(sp.I * t) for t, q in zip(theta, Pi))
    diffs = [sp.simplify((lhs - rhs).subs({q: int(i == j) for i, q in enumerate(Pi)})) for j in range(n)]
    return lhs, rhs, diffs


# --------------------------------------------------------------------------- #
# step 5-6: the detected power as a quadratic form, and its real form
# --------------------------------------------------------------------------- #
def quadratic_form(n):
    """For n petals with complex channel amplitudes h_j = x_j + i y_j, the power
    |sum_j h_j exp(i theta_j)|^2, and its real form
    sum_j |h_j|^2 + 2 sum_{j<k} [Re(h_j h_k^*) cos(theta_j - theta_k)
                                 - Im(h_j h_k^*) sin(theta_j - theta_k)].
    Returns (power, real_form, simplified difference)."""
    theta = sp.symbols(f"theta_0:{n}", real=True)
    x = sp.symbols(f"x_0:{n}", real=True)
    y = sp.symbols(f"y_0:{n}", real=True)
    h = [xi + sp.I * yi for xi, yi in zip(x, y)]
    a = sum(hj * sp.exp(sp.I * t) for hj, t in zip(h, theta))
    power = sp.expand(a * sp.conjugate(a))
    real = sum(xi ** 2 + yi ** 2 for xi, yi in zip(x, y))
    for j, k in combinations(range(n), 2):
        hh = sp.expand(h[j] * sp.conjugate(h[k]))
        real += 2 * (sp.re(hh) * sp.cos(theta[j] - theta[k]) - sp.im(hh) * sp.sin(theta[j] - theta[k]))
    diff = sp.simplify(sp.expand(sp.expand_complex(power - real), trig=True))
    return power, real, diff


def global_piston_invariance(n=3):
    """The real form depends only on phase differences: substituting
    theta_j -> theta_j + phi leaves it unchanged."""
    _, real, _ = quadratic_form(n)
    theta = sp.symbols(f"theta_0:{n}", real=True)
    phi = sp.Symbol("phi", real=True)
    shifted = real.subs({t: t + phi for t in theta}, simultaneous=True)
    return sp.simplify(shifted - real)


# --------------------------------------------------------------------------- #
# the final closed form, its constants, its Jacobian
# --------------------------------------------------------------------------- #
def closed_form(ref=N_PETALS - 1):
    """s_c = (a_c + sum_{j<k} A cos D + B sin D) / (1 + sum_{j<k} q^c cos D + q^s sin D),
    D_jk = 2 pi (p_j - p_k) / lambda, p_ref = 0. Returns (equation, pistons, lambda)."""
    _, p, lam, _ = symbols()
    pp = [0 if j == ref else p[j] for j in range(N_PETALS)]
    D = {(j, k): 2 * sp.pi * (pp[j] - pp[k]) / lam for j, k in PAIRS}
    lbl = lambda j: "R" if j == ref else str(j)
    A = {jk: sp.Symbol(f"A_{{c,{lbl(jk[0])}{lbl(jk[1])}}}") for jk in PAIRS}
    B = {jk: sp.Symbol(f"B_{{c,{lbl(jk[0])}{lbl(jk[1])}}}") for jk in PAIRS}
    qc = {jk: sp.Symbol(f"q^c_{{{lbl(jk[0])}{lbl(jk[1])}}}") for jk in PAIRS}
    qs = {jk: sp.Symbol(f"q^s_{{{lbl(jk[0])}{lbl(jk[1])}}}") for jk in PAIRS}
    num = sp.Symbol("a_c") + sum(A[jk] * sp.cos(D[jk]) + B[jk] * sp.sin(D[jk]) for jk in PAIRS)
    den = 1 + sum(qc[jk] * sp.cos(D[jk]) + qs[jk] * sp.sin(D[jk]) for jk in PAIRS)
    return sp.Eq(sp.Symbol("s_c"), num / den), [p[j] for j in range(N_PETALS) if j != ref], lam


def count_constants(n_cores=19, n_petals=N_PETALS):
    """sympy expression and value of the number of constants of the closed form."""
    nc, npet = sp.Symbol("n_cores"), sp.Symbol("n_petals")
    pairs = npet * (npet - 1) / 2
    expr = nc * (1 + 2 * pairs) + 2 * pairs
    return expr, expr.subs({nc: n_cores, npet: n_petals})


# --------------------------------------------------------------------------- #
# first principles: the constants from cbeam, with no fitting
# --------------------------------------------------------------------------- #
def petal_channel_amplitudes(p, ifunc, geom, prop12=None):
    """(6, n_cores) complex channel amplitudes h_jc: the field of petal j alone
    (unit amplitude on that petal, zero elsewhere) through the zero-padded FFT,
    projected on the lantern's input modes (without normalisation), propagated
    by cbeam and expressed in the output channel basis, divided by the
    focal-plane norm -- the convention of the spectral dataset, whose channel
    powers are fractions of the incident power. Then a configuration with petal
    phasors z_j gives the channel amplitudes sum_j z_j h_jc.

    Returns (h, g, prop12): g (6, n_modes) are the unnormalised input modal
    coefficients of the petal fields, whose quadratic form is the coupled power
    (the normalisation of the camera signals)."""
    from .simulate import bp, CBEAM_EXAMPLES
    if prop12 is None:
        with contextlib.chdir(CBEAM_EXAMPLES):
            prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
    pipe = bp.BatchPropagationPipeline(prop12, p, ifunc)
    fg, proj = pipe._field_gen_np, pipe._modal_proj_np
    pm = np.asarray(geom["petal_map"])
    E = np.zeros((N_PETALS, fg.grid_size, fg.grid_size), complex)
    for j in range(N_PETALS):
        E[j][pm == j] = 1.0
    E_mesh = fg.resample_to_mesh(fg.apply_ef_to_lantern(E))
    g = E_mesh @ proj.projection_matrix.T                  # unnormalised modal coefficients (6, n_modes)
    _, _, uf = prop12.propagate(g)
    h = np.stack([prop12.to_channel_basis(uf[j]) for j in range(N_PETALS)])
    return h / fg.focal_plane_norm, g / fg.focal_plane_norm, prop12


def channel_powers_from_amplitudes(h, P, wavelength_nm, ref=N_PETALS - 1):
    """(n, n_cores) channel powers |sum_j exp(i theta_j) h_jc|^2 of pistons P (n, 5)."""
    full = np.insert(np.asarray(P, float), ref, 0.0, axis=1)
    z = np.exp(2j * np.pi * full / wavelength_nm)
    return np.abs(z @ h) ** 2


def constants_from_amplitudes(h, ref=N_PETALS - 1):
    """(a, A, B) of the numerator form from the amplitudes: a_c = sum_j |h_jc|^2,
    A_c,jk = 2 Re(h_jc h_kc^*), B_c,jk = -2 Im(h_jc h_kc^*), in the order of
    ml_reconstruct.forward.PAIRS."""
    a = np.sum(np.abs(h) ** 2, axis=0)
    hh = np.array([h[j] * np.conj(h[k]) for j, k in PAIRS])     # (15, n_cores)
    return a, 2 * hh.real.T, -2 * hh.imag.T


def denominator_from_modes(g, ref=N_PETALS - 1):
    """(q^c, q^s) of the normalisation 1 + q . z, from the input modal
    coefficients g (6, n_modes) of the petal fields: the coupled power is
    sum_jk z_j z_k^* G_jk with G_jk = g_j . g_k^*, divided by its constant
    term sum_j |g_j|^2."""
    G0 = np.sum(np.abs(g) ** 2)
    G = np.array([np.sum(g[j] * np.conj(g[k])) for j, k in PAIRS])
    return 2 * G.real / G0, -2 * G.imag / G0


def fft_linearity_check(p, ifunc, geom, pistons, ref=N_PETALS - 1):
    """Largest relative difference between the focal-plane field of a piston
    configuration (computed as batch_pipeline does) and the phasor-weighted sum
    of the six single-petal fields: zero up to rounding, by linearity of the FFT."""
    from .simulate import bp
    fg = bp.IncidentFieldGenerator(p, ifunc)
    coeff = np.zeros((1, fg.num_modes)); coeff[0, :5] = pistons
    E_cfg = fg.apply_ef_to_lantern(fg.generate_field_profiles_batch(coeff))[0]
    pm = np.asarray(geom["petal_map"])
    Ej = np.zeros((N_PETALS, fg.grid_size, fg.grid_size), complex)
    for j in range(N_PETALS):
        Ej[j][pm == j] = 1.0
    Ej = fg.apply_ef_to_lantern(Ej)
    z = np.exp(2j * np.pi * np.insert(np.asarray(pistons, float), ref, 0.0) / p["wavelength_nm"])
    return float(np.abs(E_cfg - np.tensordot(z, Ej, 1)).max() / np.abs(E_cfg).max())


def jacobian_small(n_free=2):
    """The closed form for n_free free petals plus the reference (a small case,
    to display legibly), and its derivatives with respect to the pistons."""
    p = sp.symbols(f"p_0:{n_free}", real=True)
    lam = sp.Symbol("lambda", positive=True)
    pp = list(p) + [0]
    pairs = list(combinations(range(n_free + 1), 2))
    lbl = lambda j: "R" if j == n_free else str(j)
    D = {jk: 2 * sp.pi * (pp[jk[0]] - pp[jk[1]]) / lam for jk in pairs}
    num = sp.Symbol("a") + sum(sp.Symbol(f"A_{{{lbl(j)}{lbl(k)}}}") * sp.cos(D[(j, k)])
                               + sp.Symbol(f"B_{{{lbl(j)}{lbl(k)}}}") * sp.sin(D[(j, k)]) for j, k in pairs)
    den = 1 + sum(sp.Symbol(f"q^c_{{{lbl(j)}{lbl(k)}}}") * sp.cos(D[(j, k)])
                  + sp.Symbol(f"q^s_{{{lbl(j)}{lbl(k)}}}") * sp.sin(D[(j, k)]) for j, k in pairs)
    s = num / den
    return s, p, [sp.simplify(sp.diff(s, pi)) for pi in p], (num, den)
