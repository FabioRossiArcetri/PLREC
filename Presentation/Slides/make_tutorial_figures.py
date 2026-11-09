#!/usr/bin/env python
"""Figures for the tutorial deck (tutorial.tex) that are drawn straight from a
lantern's cached cbeam characterization -- what the simulation stores, shown
in the symbols of the slides:

    fem_mesh         the finite-element mesh of the cross-section (z = 0)
    modes_along_z    |xi_n|^2 of a few supermodes at four positions along the taper
    zgrid            the adaptive z grid: step size vs z (zvals)
    adiabaticity     max |kappa_mn| / |beta_n - beta_m| along z (cplcoeffs, eigenvalues)
    powers_along_z   |c_n(z)|^2 of a flat-wavefront launch (the coupled-mode ODE)
    fill_scan        coupling and sensing vs PSF size (scripts/fill_scan.py results)

    python make_tutorial_figures.py [lantern]       # default pl19_1200

Drawn at their size on the slide (slidefig.RC fonts), written to figures/slides/tut_*.
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, os.pardir, os.pardir)))
sys.path.insert(0, HERE)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
LANTERN = sys.argv[1] if len(sys.argv) > 1 else "pl19_1200"
os.environ["CBEAM_LANTERN"] = LANTERN
from ml_reconstruct.simulate import bp, CBEAM_EXAMPLES     # noqa: E402
import slidefig                                            # noqa: E402

plt.rcParams.update(slidefig.RC)
CM = slidefig.CM


def figure(w, h, **kw):
    return plt.subplots(figsize=(w * CM, h * CM), layout="constrained", **kw)


def save(fig, name):
    slidefig.export(fig, f"tut_{name}", native=True)
    plt.close(fig)


quiet = contextlib.redirect_stdout(io.StringIO())
p = bp.get_simulation_parameters(lantern=LANTERN)
with quiet, contextlib.chdir(CBEAM_EXAMPLES):
    prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
front, back = prop12.props if hasattr(prop12, "props") else prop12.propagators
k0 = 2 * np.pi / p["wl"]                                  # 1/um
L = p["z_ex"]
s = lambda z: (p["taper_factor"] - 1) / L * z + 1          # noqa: E731  (cbeam's linear taper)

# ------------------------------------------------------------------ FEM mesh at z = 0
mesh = front.mesh
pts = np.asarray(mesh.points)[:, :2]
tri = np.asarray(mesh.cells[1].data)[:, :3]
fig, axes = figure(11.0, 4.4, ncols=2)
for ax, lim, title in ((axes[0], p["rjack"] * 1.02, "whole cross-section"),
                       (axes[1], p["rclad"] * 0.55, "zoom: cores (z = 0)")):
    ax.triplot(Triangulation(pts[:, 0], pts[:, 1], tri), lw=0.15, color="#3a6ea5")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(title); ax.set_xlabel(r"x ($\mu$m)")
axes[0].set_ylabel(r"y ($\mu$m)")
fig.suptitle(f"{len(pts):,} nodes, {len(tri):,} triangles (quadratic elements)")
save(fig, "fem_mesh")

# ------------------------------------------------------------------ supermodes along z
zs_show = [0.0, 0.25 * L, 0.5 * L, L]
modes_show = [0, 1, 5, 14]
# their names at the multimode end (z = 0), read from the fields
LP0 = slidefig.lp_names({"points": pts, "modes": np.asarray(front.vs)[0]}, modes_show, p["rclad"],
                        all_modes=range(p["n_output_positions"]))   # same names as the 16-mode figure
fig, axes = figure(7.6, 5.7, nrows=len(modes_show), ncols=len(zs_show))
for j, z in enumerate(zs_show):
    pr = front if z <= front.zs[-1] else back
    iz = int(np.argmin(np.abs(np.asarray(pr.zs) - z)))
    v = np.asarray(pr.vs)[iz]
    xy = pts * s(z)                                      # the z = 0 mesh scaled by s(z)
    T = Triangulation(xy[:, 0], xy[:, 1], tri)
    half = p["rclad"] * s(z) * 0.62
    for i, n in enumerate(modes_show):
        ax = axes[i, j]
        I = np.abs(v[n]) ** 2
        ax.tripcolor(T, I / I.max(), shading="gouraud", cmap="magma", vmin=0, vmax=1, rasterized=True)
        ax.set_xlim(-half, half); ax.set_ylim(-half, half); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_title(f"z = {z / 1e3:.0f} mm" + chr(10) + f"scale x{s(z):.1f}")
        if j == 0:
            ax.set_ylabel(f"mode {n}" + chr(10) + LP0[n])
save(fig, "modes_along_z")

# ------------------------------------------------------------------ adaptive z grid
fig, ax = figure(7.0, 4.4)
for pr, lab, c in ((front, "front half (modes change fast)", "C0"), (back, "back half", "C1")):
    z = np.asarray(pr.zs)
    ax.semilogy(z[1:] / 1e3, np.diff(z), ".-", ms=2, lw=0.6, color=c, label=f"{lab}: {len(z)} positions")
ax.set_xlabel("z (mm)"); ax.set_ylabel(r"step $\Delta z$ ($\mu$m)")
ax.legend(loc="lower right")
save(fig, "zgrid")

# ------------------------------------------------------------------ adiabaticity along z
fig, axes = figure(12.0, 4.6, ncols=2)
# the back half is one degenerate group (equal n_eff by construction): the ratio is not defined there
for pr, c in ((front, "C0"),):
    z = np.asarray(pr.zs)
    nz = np.asarray(pr.neffs)
    worst, kmax = [], []
    for zz, nn in zip(z, nz):
        kap = np.real(np.asarray(pr.get_cmat(zz)))           # kappa_mn(z), antisymmetrised
        dbeta = k0 * np.abs(nn[:, None] - nn[None, :])
        keep = (dbeta > k0 * 1e-6) & ~np.eye(len(nn), dtype=bool)
        sk = list(pr.skipped_modes)
        keep[sk, :] = False; keep[:, sk] = False
        ratio = np.where(keep, np.abs(kap) / np.where(keep, dbeta, 1), 0)
        worst.append(ratio.max()); kmax.append(np.abs(kap[keep]).max() if keep.any() else np.nan)
    axes[0].semilogy(z / 1e3, kmax, lw=0.8, color=c)
    axes[1].semilogy(z / 1e3, worst, lw=0.8, color=c)
axes[0].set_title(r"largest coupling $|\kappa_{mn}|$ ($\mu$m$^{-1}$)")
axes[1].set_title(r"adiabaticity: max $|\kappa_{mn}| / |\beta_n-\beta_m|$")
axes[1].axhline(1, color="k", ls="--", lw=0.6)
for a in axes:
    a.set_xlabel("z (mm), front half")
save(fig, "adiabaticity")

# ------------------------------------------------------------------ modal powers along z (flat wavefront)
ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
with quiet:
    pipe = bp.BatchPropagationPipeline(prop12, p, ifunc)
    u0 = np.asarray(pipe.generate_batch_modal_coefficients(np.zeros((1, 5)), use_gpu=False))[0]
    zf_, cf, uf_front = front.propagate(u0, n_save=400)
    zb_, cb, _ = back.propagate(uf_front, n_save=200)
z_all = np.concatenate([zf_, zb_]) / 1e3
P = np.abs(np.concatenate([np.asarray(cf), np.asarray(cb)])) ** 2
fig, ax = figure(12.0, 4.6)
order = np.argsort(-P[0])
for n in order[:8]:
    ax.plot(z_all, P[:, n], lw=0.9, label=f"mode {n}")
ax.plot(z_all, P.sum(1), "k--", lw=0.7, label=r"$\sum_n |c_n|^2$")
ax.set_xlabel("z (mm)"); ax.set_ylabel(r"$|c_n(z)|^2$")
ax.set_title("flat wavefront launched at z = 0: power per supermode along the taper")
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False)
save(fig, "powers_along_z")

# ------------------------------------------------------------------ PSF size scan (scripts/fill_scan.py, 2026-11-08)
fill = np.array([0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60])
coup = np.array([15.5, 42.8, 62.5, 77.1, 82.0, 82.6, 82.9])
werr = np.array([0.42, 0.34, 0.40, 0.82, 0.55, 0.45, 0.45])
fig, ax = figure(7.0, 4.4)
ax.plot(fill, coup, "o-", color="C0")
ax.set_xlabel(r"PSF fill: Airy radius / $r_\mathrm{clad}$"); ax.set_ylabel("flat-wavefront coupling (%)", color="C0")
ax2 = ax.twinx()
ax2.plot(fill, werr, "s-", color="C3")
ax2.set_ylabel(r"wrapped error (% of $\lambda$)", color="C3")
ax.axvline(0.50, color="gray", ls=":", lw=0.6)  # the fill in use
save(fig, "fill_scan")

# ------------------------------------------------------------------ LP modes of an ideal step-index guide
# The characteristic equation U J_{l-1}(U)/J_l(U) = -W K_{l-1}(W)/K_l(W), W = sqrt(V^2 - U^2),
# solved for every l and V: b = W^2/V^2 (Gloge 1971), and the radial profiles at our V.
from scipy.special import jv, kv          # noqa: E402
from scipy.optimize import brentq         # noqa: E402


def lp_roots(V, lmax=6):
    """[(U, l, m)] of the guided LP modes at normalised frequency V."""
    out = []
    for l in range(lmax + 1):
        def f(U):
            W = np.sqrt(V * V - U * U)
            return U * jv(l - 1, U) / jv(l, U) + W * kv(l - 1, W) / kv(l, W)
        # dense near U = V too: there the low-V roots (LP01 has no cut-off) sit
        Us = np.unique(np.concatenate([np.linspace(1e-4, V - 1e-7, 4000), V - np.geomspace(1e-7, 0.5 * V, 400)]))
        with np.errstate(all="ignore"):
            vals = f(Us)
        m = 0
        for a, b_, va, vb in zip(Us[:-1], Us[1:], vals[:-1], vals[1:]):
            if np.isfinite(va) and np.isfinite(vb) and va * vb < 0 and abs(va - vb) < 50:
                m += 1
                out.append((brentq(f, a, b_), l, m))
    return out


V_OURS = p["V_mmf"]
fig, axes = figure(6.4, 6.6, nrows=2)
# (a) radial profiles at our V: J_l inside the core, K_l outside, matched at r = rho
r = np.linspace(0, 2.4, 500)
for (U, l, m), c in zip(sorted(lp_roots(V_OURS))[:6], plt.cm.tab10.colors):
    W = np.sqrt(V_OURS ** 2 - U ** 2)
    with np.errstate(all="ignore"):
        prof = np.where(r < 1, jv(l, U * r), jv(l, U) / kv(l, W) * kv(l, W * np.maximum(r, 1e-9)))
    prof = prof / np.abs(prof).max()
    axes[0].plot(r, prof, color=c, label=f"LP$_{{{l}{m}}}$")
axes[0].axvspan(0, 1, color="0.92", zorder=0)
axes[0].axhline(0, color="k", lw=0.4)
axes[0].text(0.5, -0.95, "core", ha="center"); axes[0].text(1.7, -0.95, "cladding", ha="center")
axes[0].set_xlabel(r"$r/\rho$"); axes[0].set_ylabel(r"$\psi_{lm}(r)$ (normalised)")
axes[0].set_title(rf"radial profiles at $V = {V_OURS:.1f}$")
axes[0].set_ylim(-1.05, 1.05)
axes[0].set_xlim(0, 2.4)
axes[0].legend(ncol=2, loc="upper right", handlelength=1.0, columnspacing=0.8, framealpha=0.9)
# (b) the normalised propagation constant b(V) of the first LP modes
Vs = np.linspace(0.3, 10.0, 160)
curves = {}
for V in Vs:
    for U, l, m in lp_roots(V):
        curves.setdefault((l, m), []).append((V, 1 - (U / V) ** 2))
cutoffs = {}                                # labels at each mode's cut-off (coincident ones merged)
curves[(0, 1)] = [(0.0, 0.0)] + curves[(0, 1)]   # LP01: no cut-off, b -> 0 as V -> 0
for (l, m), pts_ in sorted(curves.items()):
    v_, b_ = np.array(pts_).T
    axes[1].plot(v_, b_, lw=0.8, color="C0" if l == 0 else "0.35")
    if (l, m) != (0, 1):
        cutoffs.setdefault(round(v_.min(), 1), []).append(f"{l}{m}")
for vc, names_ in cutoffs.items():
    if vc < 8.0:
        axes[1].text(vc, 0.03, "LP$_{" + ",".join(names_) + "}$", rotation=90, ha="right", va="bottom", fontsize=4.5)
axes[1].text(1.0, 0.55, "LP$_{01}$", fontsize=4.5, color="C0")
axes[1].axvline(V_OURS, color="C3", ls="--", lw=0.7)
axes[1].text(V_OURS, 1.02, f"our $V = {V_OURS:.2f}$", color="C3", ha="center", va="bottom")
axes[1].set_xlim(0, 11.2); axes[1].set_ylim(0, 1)
axes[1].set_xlabel("$V$"); axes[1].set_ylabel(r"$b = W^2/V^2$")
save(fig, "lp_modes")
print("done")
