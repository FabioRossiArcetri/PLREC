"""Exact analytical forward model of the lantern's single-wavelength response to
petal pistons, and its inversion (symbolic.ipynb).

With piston-only aberrations and monochromatic light, the field at every
output core is linear in the six petal phasors exp(i phi_j), so each output
power is a Hermitian quadratic form in them:

    N_c(p) = a_c + sum_{j<k} [A_cjk cos D_jk + B_cjk sin D_jk],   D_jk = 2 pi (p_j - p_k) / lambda

(the reference petal has p = 0). The camera signals of the single-wavelength
dataset are normalised by the coupled power, itself such a form, so

    s_c(p) = N_c(p) / (1 + q . z(p)),          z = [cos D, sin D]  (30 features)

with 19 x 31 + 30 = 619 constants, linear in the constants given p: they are
fitted by one linear least-squares solve (``fit_forward``). A reconstructor is
then the inversion of this closed form: Gauss-Newton on the 5 pistons, with
the analytic Jacobian, from several starts, keeping the lowest residual
(``invert``).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

N_PETALS = 6
PAIRS = list(combinations(range(N_PETALS), 2))
_J = np.array([a for a, _ in PAIRS])
_K = np.array([b for _, b in PAIRS])


def pair_phases(P, wavelength_nm, ref=5):
    """(n, 15) pairwise phase differences D_jk of pistons P (n, 5) (the reference petal at 0)."""
    full = np.insert(np.asarray(P, float), ref, 0.0, axis=1)
    return 2 * np.pi * (full[:, _J] - full[:, _K]) / wavelength_nm


def pair_features(P, wavelength_nm, ref=5):
    """(n, 30) features [cos D, sin D]."""
    D = pair_phases(P, wavelength_nm, ref)
    return np.concatenate([np.cos(D), np.sin(D)], axis=1)


@dataclass
class ForwardModel:
    """s(p) = (a + z(p) @ W.T) / (1 + z(p) @ q)  (q = 0: channel powers, no normalisation)."""
    a: np.ndarray                  # (n_cores,)
    W: np.ndarray                  # (n_cores, 30)
    q: np.ndarray                  # (30,)
    wavelength_nm: float
    ref: int = 5

    @property
    def n_constants(self):
        return int(self.a.size + self.W.size + np.count_nonzero(self.q))

    def __call__(self, P):
        z = pair_features(P, self.wavelength_nm, self.ref)
        return (self.a + z @ self.W.T) / (1 + z @ self.q)[:, None]

    def jacobian(self, P):
        """(n, n_cores, 5) ds/dp, analytic."""
        D = pair_phases(P, self.wavelength_nm, self.ref)
        z = np.concatenate([np.cos(D), np.sin(D)], axis=1)
        dz_dD = np.concatenate([-np.sin(D), np.cos(D)], axis=1)              # (n, 30), d z / d D of its pair
        keep = [k for k in range(N_PETALS) if k != self.ref]
        dD_dp = np.zeros((15, 5))                                           # D_jk = c (p_j - p_k)
        for i, (j, k) in enumerate(PAIRS):
            if j != self.ref: dD_dp[i, keep.index(j)] += 1
            if k != self.ref: dD_dp[i, keep.index(k)] -= 1
        dD_dp *= 2 * np.pi / self.wavelength_nm
        dz_dp = dz_dD[:, :, None] * np.concatenate([dD_dp, dD_dp])[None]   # (n, 30, 5)
        N = self.a + z @ self.W.T; Dn = 1 + z @ self.q
        dN = np.einsum("cf,nfp->ncp", self.W, dz_dp)
        dDn = np.einsum("f,nfp->np", self.q, dz_dp)
        return (dN * Dn[:, None, None] - N[:, :, None] * dDn[:, None, :]) / Dn[:, None, None] ** 2


def fit_forward(P, S, wavelength_nm, normalised=True, ref=5):
    """Least-squares fit of the forward model to pistons P (n, 5) and signals
    S (n, n_cores). normalised: the signals are divided by the coupled power
    (camera dataset); False for channel powers (q = 0)."""
    z = pair_features(P, wavelength_nm, ref)
    n, nc = S.shape
    if not normalised:
        X = np.c_[np.ones(n), z]
        coef, *_ = np.linalg.lstsq(X, S, rcond=None)
        return ForwardModel(coef[0], coef[1:].T, np.zeros(30), wavelength_nm, ref)
    # s_c (1 + q.z) = a_c + W_c.z  ->  s_c = a_c + W_c.z - s_c (q.z), linear in (a, W, q), q shared
    X = np.zeros((n * nc, nc * 31 + 30))
    for c in range(nc):
        r = slice(c * n, (c + 1) * n)
        X[r, c * 31] = 1; X[r, c * 31 + 1:(c + 1) * 31] = z; X[r, nc * 31:] = -S[:, [c]] * z
    sol, *_ = np.linalg.lstsq(X, S.T.reshape(-1), rcond=None)
    C = sol[:nc * 31].reshape(nc, 31)
    return ForwardModel(C[:, 0], C[:, 1:], sol[nc * 31:], wavelength_nm, ref)


def gauss_newton(model, S, P0, iters=12, bound_nm=None):
    """Gauss-Newton from P0 (n, 5); returns (P, residual norm per sample)."""
    P = np.array(P0, float)
    for _ in range(iters):
        r = S - model(P)
        Jm = model.jacobian(P)
        JT = np.transpose(Jm, (0, 2, 1))
        P = P + np.linalg.solve(JT @ Jm + 1e-14 * np.eye(5), JT @ r[:, :, None])[:, :, 0]
        if bound_nm is not None:
            P = np.clip(P, -bound_nm, bound_nm)
    return P, np.linalg.norm(S - model(P), axis=1)


def invert(model, S, P_init, n_starts=64, span_nm=None, iters=12, tol=None, rng=0):
    """Multi-start inversion: Gauss-Newton from P_init, then from random starts in
    +-span_nm (default +-lambda/2 of the model), keeping per sample the solution with the lowest residual. tol: a
    residual below which a sample is considered solved and not restarted (None:
    restart every sample). Returns (P, residual norm, starts used per sample)."""
    rng = np.random.default_rng(rng)
    span_nm = model.wavelength_nm / 2 if span_nm is None else span_nm
    P, res = gauss_newton(model, S, P_init, iters, bound_nm=1.5 * span_nm)
    used = np.ones(len(S), int)
    for _ in range(n_starts):
        todo = np.ones(len(S), bool) if tol is None else res > tol
        if not todo.any():
            break
        Pt, rt = gauss_newton(model, S[todo], rng.uniform(-span_nm, span_nm, (todo.sum(), 5)), iters,
                              bound_nm=1.5 * span_nm)
        idx = np.flatnonzero(todo)
        better = rt < res[idx]
        P[idx[better]], res[idx[better]] = Pt[better], rt[better]
        used[idx] += 1
    return P, res, used


def error_summary(P_hat, P_true, gross_nm=50.0):
    """RMSE per piston, and the median, 99th percentile and gross-error share of |error|."""
    e = np.abs(P_hat - P_true)
    return dict(rmse=np.sqrt(np.mean((P_hat - P_true) ** 2, axis=0)).round(3), median=np.median(e),
                p99=np.percentile(e, 99), **{f"> {gross_nm:g} nm %": 100 * np.mean(e.max(1) > gross_nm)})


def noise_scan(model, S, P, init, levels, n_starts=32, rng=0):
    """Inversion accuracy with Gaussian noise of std level x the mean signal
    added to S; init: function S -> initial pistons."""
    import pandas as pd
    g = np.random.default_rng(rng)
    rows = []
    for lev in levels:
        Sn = S + g.normal(0, lev * S.mean(), S.shape) if lev > 0 else S
        Ph, _, _ = invert(model, Sn, init(Sn), n_starts=n_starts)
        rows.append(dict(noise=lev, **error_summary(Ph, P)))
    return pd.DataFrame(rows)


def calibration_scan(P_cal, S_cal, sizes, P, S, init, wavelength_nm, noise=0.0, n_starts=32, rng=0):
    """Forward models fitted on the first n calibration samples (with noise of
    std noise x the mean signal added to them and to the test signals), and the
    resulting inversion accuracy on (P, S)."""
    import pandas as pd
    g = np.random.default_rng(rng)
    Sc = S_cal + g.normal(0, noise * S_cal.mean(), S_cal.shape) if noise else S_cal
    St = S + g.normal(0, noise * S.mean(), S.shape) if noise else S
    rows = []
    for n in sizes:
        m = fit_forward(P_cal[:n], Sc[:n], wavelength_nm)
        rel = np.sqrt(np.mean((S - m(P)) ** 2)) / S.std()
        Ph, _, _ = invert(m, St, init(St), n_starts=n_starts)
        rows.append(dict(n_calibration=n, model_rel_residual=rel, **error_summary(Ph, P)))
    return pd.DataFrame(rows)


def plot_coefficients(model, pair_labels=None):
    """|A + i B| of each (output core, petal pair): how strongly each core's
    power interferes the two petals of the pair (the forward model's
    numerator, relative to the core's constant term a_c)."""
    import matplotlib.pyplot as plt
    A, B = model.W[:, :15], model.W[:, 15:]
    M = np.abs(A + 1j * B) / np.abs(model.a)[:, None]
    labels = pair_labels or [f"{'R' if j == model.ref else j}-{'R' if k == model.ref else k}" for j, k in PAIRS]
    plt.figure(figsize=(11, 8))
    plt.imshow(M.T, aspect="equal", cmap="viridis")
    plt.colorbar(label="|A + iB| / a  (interference strength)", fraction=0.035)
    plt.yticks(range(15), labels, fontsize=8); plt.xticks(range(M.shape[0]))
    plt.xlabel("output core"); plt.ylabel("petal pair")
    plt.title("forward model: interference of each petal pair in each output core")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# convergence analysis and a staged solver
# --------------------------------------------------------------------------- #
def gn_trace(model, S, P0, P_true, iters=15, bound_nm=450.0):
    """(iters + 1, n) largest |error| over the pistons after each Gauss-Newton
    iteration from P0 (row 0: the start)."""
    P = np.array(P0, float)
    out = [np.abs(P - P_true).max(1)]
    for _ in range(iters):
        P, _ = gauss_newton(model, S, P, 1, bound_nm)
        out.append(np.abs(P - P_true).max(1))
    return np.array(out)


def convergence_table(traces, thresholds_nm=(1.0, 0.01), iterations=(1, 2, 3, 5, 10)):
    """Share of cases whose largest error is below each threshold after at most
    each number of iterations, for {start: gn_trace}."""
    import pandas as pd
    rows = []
    for name, H in traces.items():
        for thr in thresholds_nm:
            ok = H < thr
            first = np.where(ok.any(0), np.argmax(ok, axis=0), np.inf)
            rows.append(dict(start=name, threshold_nm=thr,
                             **{f"<= {n} it": 100 * np.mean(first <= n) for n in iterations},
                             **{f"not in {len(H) - 1} it": 100 * np.mean(np.isinf(first))}))
    return pd.DataFrame(rows)


def solve_staged(model, S, P0, T, tol, n_first=5, n_cand=8, n_restart=8, max_restarts=500, span_nm=None, rng=0):
    """The reconstructor as a procedure: n_first Gauss-Newton iterations from P0;
    for the cases whose residual is still above tol, n_cand iterations from the
    twin T p, the sign flip -p, -T p and the flat wavefront of the estimate; then
    random restarts (n_restart iterations each) for whatever is left. Returns
    (P, residual, total iterations per case, stage that solved each case:
    0 first, 1 candidates, 2 random restarts, 3 not solved). span_nm: range of the
    random restarts (default +-lambda/2 of the model)."""
    g = np.random.default_rng(rng)
    span_nm = model.wavelength_nm / 2 if span_nm is None else span_nm
    P, r = gauss_newton(model, S, P0, n_first, 1.5 * span_nm)
    iters, stage = np.full(len(S), n_first), np.where(r <= tol, 0, 3)
    bad = np.flatnonzero(r > tol)
    if len(bad):
        base = P[bad].copy()
        for cand in (base @ T.T, -base, -(base @ T.T), np.zeros_like(base)):
            todo = bad[r[bad] > tol]
            if not len(todo):
                break
            sel = np.isin(bad, todo)
            Pc, rc = gauss_newton(model, S[todo], cand[sel], n_cand, 1.5 * span_nm)
            iters[todo] += n_cand
            better = rc < r[todo]
            P[todo[better]], r[todo[better]] = Pc[better], rc[better]
        stage[bad[r[bad] <= tol]] = 1
    for _ in range(max_restarts):
        todo = np.flatnonzero(r > tol)
        if not len(todo):
            break
        Pc, rc = gauss_newton(model, S[todo], g.uniform(-span_nm, span_nm, (len(todo), 5)), n_restart, 1.5 * span_nm)
        iters[todo] += n_restart
        better = rc < r[todo]
        P[todo[better]], r[todo[better]] = Pc[better], rc[better]
        stage[todo[rc <= tol]] = 2
    return P, r, iters, stage


def time_single_case(model, s, p0, n_iter=5, repeat=200):
    """Wall time [us] of n_iter Gauss-Newton iterations for one case (numpy, one thread)."""
    import time
    s, p0 = np.atleast_2d(s), np.atleast_2d(p0)
    t = time.perf_counter()
    for _ in range(repeat):
        gauss_newton(model, s, p0, n_iter)
    return (time.perf_counter() - t) / repeat * 1e6


# --------------------------------------------------------------------------- #
# the closed form, with sympy
# --------------------------------------------------------------------------- #
def sympy_generic():
    """The forward model as a sympy expression with symbolic constants:
    (s_c, the pistons p_0..p_4, lambda)."""
    import sympy as sp
    p = sp.symbols("p_0:5"); lam = sp.Symbol("lambda", positive=True)
    ph = list(p) + [0]
    D = {(j, k): 2 * sp.pi * (ph[j] - ph[k]) / lam for j, k in PAIRS}
    A = {jk: sp.Symbol(f"A_{{c,{jk[0]}{jk[1]}}}") for jk in PAIRS}
    B = {jk: sp.Symbol(f"B_{{c,{jk[0]}{jk[1]}}}") for jk in PAIRS}
    qa = {jk: sp.Symbol(f"q^c_{{{jk[0]}{jk[1]}}}") for jk in PAIRS}
    qb = {jk: sp.Symbol(f"q^s_{{{jk[0]}{jk[1]}}}") for jk in PAIRS}
    num = sp.Symbol("a_c") + sum(A[jk] * sp.cos(D[jk]) + B[jk] * sp.sin(D[jk]) for jk in PAIRS)
    den = 1 + sum(qa[jk] * sp.cos(D[jk]) + qb[jk] * sp.sin(D[jk]) for jk in PAIRS)
    return sp.Eq(sp.Symbol("s_c"), num / den), p, lam


def sympy_forward(model, core, n_terms=None, digits=3):
    """The fitted forward model of one output core as a sympy expression in the
    pistons p_0..p_4 [nm], with numeric constants (relative to the core's
    constant term); n_terms keeps only the largest numerator / denominator terms."""
    import sympy as sp
    p = sp.symbols("p_0:5"); ph = list(p) + [0]
    lam = model.wavelength_nm
    D = [2 * sp.pi * (ph[j] - ph[k]) / lam for j, k in PAIRS]
    trig = [sp.cos(d) for d in D] + [sp.sin(d) for d in D]
    a = model.a[core]
    w, q = model.W[core] / a, model.q

    def terms(coef):
        idx = np.argsort(-np.abs(coef))[:n_terms] if n_terms else np.arange(len(coef))
        return sum(sp.Float(round(float(coef[i]), digits), digits) * trig[i] for i in sorted(idx)
                   if round(float(coef[i]), digits) != 0)
    num = 1 + terms(w)
    den = 1 + terms(q)
    return sp.Eq(sp.Symbol(f"s_{{{core}}}"), sp.Mul(sp.Float(float(a), max(digits, 4)), num / den, evaluate=False))


def plot_convergence(traces, thresholds_nm=(1.0, 0.01)):
    """Share of cases below each error threshold vs. the Gauss-Newton
    iteration, one line per start ({name: gn_trace}), and the median and
    99.9th percentile of the error."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i, (name, H) in enumerate(traces.items()):
        it = np.arange(len(H))
        for thr, ls in zip(thresholds_nm, ("-", "--")):
            ax[0].plot(it, 100 * (H < thr).mean(1), ls, color=plt.cm.tab10(i), label=f"{name}, < {thr:g} nm")
        ax[1].plot(it, np.median(H, 1), "-", color=plt.cm.tab10(i), label=f"{name}, median")
        ax[1].plot(it, np.percentile(H, 99.9, axis=1), ":", color=plt.cm.tab10(i), label=f"{name}, 99.9%")
    ax[0].set_ylabel("cases (%)"); ax[0].set_ylim(0, 101)
    ax[1].set_yscale("log"); ax[1].set_ylabel("largest piston error (nm)")
    for a in ax:
        a.set_xlabel("Gauss-Newton iteration"); a.grid(alpha=0.3, which="both"); a.legend(fontsize=7)
    plt.tight_layout(); plt.show()


def plot_noise_scan(scans):
    """RMSE (mean over pistons) and 99th percentile of |error| vs. noise level,
    one line per {label: noise_scan table}."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 4.2))
    for i, (label, df) in enumerate(scans.items()):
        d = df[df.noise > 0]
        plt.loglog(d.noise, [np.mean(r) for r in d.rmse], "o-", color=plt.cm.tab10(i), label=f"{label}: RMSE")
        plt.loglog(d.noise, d.p99, "s:", color=plt.cm.tab10(i), label=f"{label}: 99th percentile")
    plt.xlabel("noise std / mean signal"); plt.ylabel("piston error (nm)")
    plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=8); plt.title("analytical reconstructor vs. noise")
    plt.tight_layout(); plt.show()


def unrolled_newton_table(model, S, P, starts, steps=(0, 1, 2, 3)):
    """A fixed number of Gauss-Newton steps from explicit first estimates: an
    explicit formula (no stopping rule, no restarts). {start name: estimates}
    -> RMSE (mean over pistons), median |error| and share of cases within 1 nm
    after each number of steps."""
    import pandas as pd
    rows = []
    for name, p0 in starts.items():
        for n in steps:
            p = p0 if n == 0 else gauss_newton(model, S, p0, n, bound_nm=450.0)[0]
            e = np.abs(p - P)
            rows.append(dict(start=name, steps=n, rmse=np.sqrt(np.mean(e ** 2)), median=np.median(e),
                             **{"within 1 nm %": 100 * np.mean(e.max(1) < 1)}))
    return pd.DataFrame(rows).pivot(index="start", columns="steps", values=["rmse", "median", "within 1 nm %"])
