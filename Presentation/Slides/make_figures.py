#!/usr/bin/env python
"""Slide figures from the cbeam simulation of a lantern preset (default pl19_800).

    python make_figures.py [lantern]        # writes figures/slides/<lantern>_*.pdf/png

Run with the plrec environment; needs the lantern's characterization caches in
../../../cbeam/examples/data (CBEAM_EXAMPLES). Every figure is built from the
same objects the analysis uses (batch_pipeline, ml_reconstruct), so the slides
show what the simulation actually does.

The figures are drawn at their size on the slide (cm, SIZE below), with all
text at slidefig.FONT_PT, and included at natural size by main_v2.tex: the
explanations are in the slide text and captions, so titles are short.
"""
import contextlib
import io
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
PLREC = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
sys.path.insert(0, PLREC)
os.environ.setdefault("CBEAM_BACKEND", "numpy")
LANTERN = sys.argv[1] if len(sys.argv) > 1 else "pl19_800"
os.environ["CBEAM_LANTERN"] = LANTERN

from ml_reconstruct.simulate import bp, CBEAM_EXAMPLES          # noqa: E402
from ml_reconstruct.petals import petal_geometry                # noqa: E402
sys.path.insert(0, HERE)
import slidefig                                                 # noqa: E402

REF = LANTERN == "pl19_1200"           # the reference lantern is in the main talk, the others in the appendix
# figure sizes on the slide (cm): main talk (reference) / appendix (1200 nm lanterns)
SIZE = {"geometry": (8.6, 4.6) if REF else (7.0, 4.4), "neff_vs_z": (9.5, 5.0) if REF else (7.0, 4.4),
        "modes_input": (6.6, 6.2), "psf_coupling": (14.0, 5.8) if REF else (14.0, 6.2),
        "transfer_matrix": (9.6, 4.4), "output_face": (14.0, 5.0), "modes_in_pupil": (7.6, 3.2),
        "pupil_petals": (5.2, 5.0)}
plt.rcParams.update(slidefig.RC)


def figure(name, **kw):
    w, h = SIZE[name]
    return plt.subplots(figsize=(w * slidefig.CM, h * slidefig.CM), layout="constrained", **kw)


def save(fig, name):
    slidefig.export(fig, f"{LANTERN}_{name}", native=True)
    plt.close(fig)


def lam_frac(k):
    return {1: r"$\lambda/4$", 2: r"$\lambda/2$"}[k]


quiet = contextlib.redirect_stdout(io.StringIO())
p = bp.get_simulation_parameters(lantern=LANTERN)
ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
with quiet, contextlib.chdir(CBEAM_EXAMPLES):
    prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
    pipe = bp.BatchPropagationPipeline(prop12, p, ifunc)
nports, wl = p["n_output_positions"], p["wavelength_nm"]
tf = p["taper_factor"]

# ------------------------------------------------------------------ geometry
fig, axes = figure("geometry", ncols=2)
for ax, s, title in [(axes[0], 1.0, "multimode end"), (axes[1], tf, f"single-mode end (x{tf:g})")]:
    ax.add_patch(Circle((0, 0), p["rjack"] * s, color="#c9d6e8", label="jacket (F-doped)"))
    ax.add_patch(Circle((0, 0), p["rclad"] * s, color="#6b8fc7", label="cladding (silica)"))
    for (x, y) in p["output_positions"]:
        ax.add_patch(Circle((x * s, y * s), p["rcore"] * s * (1 if s > 1 else 3), color="#d1495b",
                            label="cores" if (x, y) == tuple(p["output_positions"][0]) else None))
    lim = p["rjack"] * s * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_xlabel(r"x ($\mu$m)"); ax.set_title(title)
    ax.locator_params(nbins=3)
axes[0].text(0, -p["rjack"] * 0.95, "cores 3x enlarged", ha="center", va="bottom")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False, handlelength=1.0, columnspacing=1.0)
save(fig, "geometry")

# ------------------------------------------------------------- n_eff(z) along the taper
fig, ax = figure("neff_vs_z")
for prop in prop12.props if hasattr(prop12, "props") else prop12.propagators:
    z = np.asarray(prop.zs) / 1e3
    n = np.real(np.asarray(prop.neffs))
    ax.plot(z, n, lw=0.8)
ax.axhline(p["nclad"], color="k", ls="--", lw=0.6)
ax.text(1, p["nclad"], "cladding index", va="bottom")
ax.axhline(p["njack"], color="gray", ls=":", lw=0.6)
ax.text(1, p["njack"], "jacket index", va="bottom")
ax.set_xlabel("z along the taper (mm)"); ax.set_ylabel(r"$n_\mathrm{eff}$ of the supermodes")
save(fig, "neff_vs_z")

# ------------------------------------------------------------- mode profiles at the multimode end
from scipy.interpolate import griddata                         # noqa: E402
win, wout = pipe.wvg_props_input, pipe.wvg_props_output


def mode_maps(props, idx, half):
    pts = props["points"]; g = np.linspace(-half, half, 160)
    X, Y = np.meshgrid(g, g)
    return [griddata(pts, np.abs(np.asarray(props["modes"])[k]) ** 2, (X, Y), fill_value=0) for k in idx], half


lp_name = slidefig.lp_name
nshow = min(16, nports)              # 4 x 4; only modes that reach a core (7 for the 7-port lantern)
# degenerate pairs: LP_lma / LP_lmb, counted over the guided modes (one per core; the
# extra near-cut-off mode is not readable)
LP = slidefig.lp_names(win, range(nshow), p["rclad"], all_modes=range(nports))
maps_in, h_in = mode_maps(win, range(nshow), p["rclad"] * 1.4)
fig, axes = figure("modes_input", nrows=4, ncols=4)
for ax in axes.flat[nshow:]:
    ax.axis("off")
for k, (ax, m) in enumerate(zip(axes.flat, maps_in)):
    ax.imshow(m, extent=[-h_in, h_in, -h_in, h_in], origin="lower", cmap="magma")
    ax.add_patch(Circle((0, 0), p["rclad"], fill=False, ec="w", lw=0.4, ls="--"))
    ax.set_title(f"{k}: {LP[k]}", pad=2); ax.axis("off")
save(fig, "modes_input")

# ------------------------------------------------------------- PSF on the lantern face and modal coupling
ref = petal_geometry(np.asarray(ifunc.mask_inf_func) > 0, np.asarray(ifunc.influence_function), 5)
fg = pipe._field_gen_np
coeffs = np.zeros((3, fg.num_modes)); coeffs[1, 0] = wl / 4; coeffs[2, 0] = wl / 2
# petal 0 at lambda/4 and at lambda/2: the edge of the single-wavelength range (+-lambda/2)
row_labels = ["flat", "petal 0\n" + rf"$\lambda/4$ = {wl / 4:.0f} nm", "petal 0\n" + rf"$\lambda/2$ = {wl / 2:.0f} nm"]
labels = ["flat wavefront", rf"petal 0: $\lambda/4$ ({wl / 4:.0f} nm)", rf"petal 0: $\lambda/2$ ({wl / 2:.0f} nm)"]
E_foc = fg.apply_ef_to_lantern(fg.generate_field_profiles_batch(coeffs))
_, padded = fg._pad_geometry(); c = padded // 2; ps = p["pixel_scale_um"]
half_px = int(p["rclad"] * 1.6 / ps)
crop = E_foc[:, c - half_px:c + half_px, c - half_px:c + half_px]
ext = [-half_px * ps, half_px * ps, -half_px * ps, half_px * ps]
with quiet:
    u0, eff = pipe.generate_batch_modal_coefficients(coeffs, use_gpu=False, return_coupling_efficiency=True)
u0 = np.asarray(u0) * np.asarray(eff)[:, None]
# the lantern output: power in each single-mode core (share of the coupled light, P / P.sum)
with quiet:
    u0n = pipe.generate_batch_modal_coefficients(coeffs, use_gpu=False)
    uf, _, _ = pipe.propagate_batch(u0n)
P = np.abs(np.stack([prop12.to_channel_basis(np.asarray(uf)[k]) for k in range(len(uf))])[:, :nports]) ** 2
# ... as a fraction of the incident power, like the mode powers (lossless: same sums)
P_inc = P / P.sum(1, keepdims=True) * (np.abs(u0) ** 2).sum(1, keepdims=True)

fig, axes = figure("psf_coupling", nrows=3, ncols=5, gridspec_kw={"width_ratios": [0.9, 0.9, 1.4, 1.4, 1.3]})
xy = np.asarray(p["output_positions"]) * tf
rad = 0.47 * np.min(np.hypot(*(xy[1:] - xy[0]).T))
lim_face = np.abs(xy).max() + 1.05 * rad          # the core pattern fills its panel
pm = np.where(ref["petal_map"] < 0, np.nan, 0.0)
ymax = 1.15 * (np.abs(u0) ** 2).max()
ymax_out = 1.15 * P_inc.max()
for r in range(3):
    opd = pm.copy(); opd[ref["petal_map"] == 0] = coeffs[r, 0]
    a = axes[r, 0]
    a.imshow(opd, cmap="RdBu_r", vmin=-wl / 2, vmax=wl / 2)
    a.set_xticks([]); a.set_yticks([]); a.set_ylabel(row_labels[r])
    for sp in a.spines.values():
        sp.set_visible(False)
    I = np.abs(crop[r]) ** 2
    a = axes[r, 1]
    a.imshow(np.log10(I / np.abs(crop[0]).max() ** 2 + 1e-5), extent=ext, origin="lower", cmap="inferno",
             vmin=-4, vmax=0)
    a.add_patch(Circle((0, 0), p["rclad"], fill=False, ec="c", ls="--", lw=0.5))
    a.plot(*np.asarray(p["output_positions"]).T, "c.", ms=1.0)
    a.set_xticks([]); a.set_yticks([])
    a = axes[r, 2]
    pw = np.abs(u0[r]) ** 2
    a.bar(np.arange(u0.shape[1]), pw, color="#3a6ea5", width=0.8)
    a.set_ylim(0, ymax); a.locator_params(axis="y", nbins=2)
    a.text(0.03, 0.92, f"sum {pw.sum():.0%}", transform=a.transAxes, ha="left", va="top")
    if r < 2:
        a.set_xticklabels([])
    a = axes[r, 3]
    a.bar(np.arange(nports), P_inc[r], color="#c0504d", width=0.8)
    a.set_ylim(0, ymax_out); a.locator_params(axis="y", nbins=2)
    if r < 2:
        a.set_xticklabels([])
    a = axes[r, 4]                         # the output face: share of the coupled light per core (%)
    for (x, y), v in zip(xy, P[r] / P[r].sum()):
        a.add_patch(Circle((x, y), rad, color=plt.cm.magma(v / 0.25)))
        a.text(x, y, f"{100 * v:.0f}", ha="center", va="center", fontsize=4.5, color="w" if v < 0.15 else "k")
    a.set_xlim(-lim_face, lim_face); a.set_ylim(-lim_face, lim_face); a.set_aspect("equal"); a.axis("off")
axes[0, 0].set_title("pupil"); axes[0, 1].set_title("PSF (log)"); axes[0, 2].set_title("power per mode")
axes[0, 3].set_title("output: power per core"); axes[0, 4].set_title("output face (%)")
axes[2, 2].set_xlabel("mode index"); axes[2, 3].set_xlabel("core index")
for a in axes[:, 2:4].flat:
    a.xaxis.set_major_locator(MaxNLocator(integer=True))
save(fig, "psf_coupling")

# ------------------------------------------------------------- transfer matrix
with quiet:
    A = []
    for n in range(prop12.Nmax if hasattr(prop12, "Nmax") else p["n_output_positions"] + 1):
        e = np.zeros(p["n_output_positions"] + 1, complex); e[n] = 1
        _, _, uf = prop12.propagate(e)
        A.append(prop12.to_channel_basis(np.asarray(uf))[:nports])
A = np.array(A).T                                               # (ports, input modes)
fig, axes = figure("transfer_matrix", ncols=2)
im = axes[0].imshow(np.abs(A) ** 2, cmap="viridis"); fig.colorbar(im, ax=axes[0], shrink=0.9)
axes[0].set_title(r"$|A_{kn}|^2$"); axes[0].set_xlabel("input mode n"); axes[0].set_ylabel("output port k")
im = axes[1].imshow(np.angle(A), cmap="twilight", vmin=-np.pi, vmax=np.pi)
cb = fig.colorbar(im, ax=axes[1], shrink=0.9, ticks=[-np.pi, 0, np.pi]); cb.ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])
axes[1].set_title(r"$\arg A_{kn}$"); axes[1].set_xlabel("input mode n")
save(fig, "transfer_matrix")
np.save(os.path.join(HERE, "figures", "sim", f"{LANTERN}_transfer_matrix.npy"), A)

# ------------------------------------------------------------- output face
lim = np.abs(xy).max() * 1.3 + 10
fig, axes = figure("output_face", ncols=3)
for r, ax in enumerate(axes):
    for (x, y), v in zip(xy, P[r] / P[r].sum()):
        ax.add_patch(Circle((x, y), 0.45 * np.min(np.hypot(*(xy[1:] - xy[0]).T)), color=plt.cm.magma(v / 0.25)))
        ax.text(x, y, f"{100 * v:.0f}", ha="center", va="center", color="w" if v < 0.15 else "k")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
    ax.set_title(labels[r]); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
save(fig, "output_face")

# ------------------------------------------------------------- guided modes seen from the pupil
# u_n = <psi_n|E> = <Psi_n|P exp(i phi)> (Parseval): Psi_n is mode n carried back to
# the pupil plane. Sample each mode on the focal-plane FFT grid and inverse-FFT it.
from scipy.interpolate import LinearNDInterpolator                # noqa: E402
N = fg.grid_size; pad, padded = fg._pad_geometry(); c0 = padded // 2
half = int(np.ceil(np.abs(win["points"]).max() / ps)) + 2
gx = (np.arange(-half, half + 1)) * ps
GX, GY = np.meshgrid(gx, gx)
modes_in = np.asarray(win["modes"])
# LP01 and the two modes that take most of the flat-wavefront light
show = [0] + [int(k) for k in np.argsort(-np.abs(u0[0, 1:]) ** 2)[:2] + 1]
interp = LinearNDInterpolator(win["points"], modes_in[show].T, fill_value=0)
pup_mask = fg.mask_np
fig, axes = figure("modes_in_pupil", ncols=len(show))
for ax, n, vals in zip(axes, show, np.moveaxis(interp(GX, GY), -1, 0)):
    F = np.zeros((padded, padded), complex)
    F[c0 - half:c0 + half + 1, c0 - half:c0 + half + 1] = vals
    Psi = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(F)))[pad:pad + N, pad:pad + N]
    I = np.abs(Psi) ** 2; I /= I.max()
    ax.imshow(I, cmap="magma", origin="upper")
    ax.contour(pup_mask.astype(float), levels=[0.5], colors="c", linewidths=0.4)
    frac = (I * pup_mask).sum() / I.sum()
    ax.set_title(f"mode {n}: {LP.get(n) or lp_name(win, n, p['rclad'])}\n{100 * frac:.0f}% on the pupil"); ax.axis("off")
save(fig, "modes_in_pupil")

# ------------------------------------------------------------- the ANDES pupil and its petals
fig, ax = figure("pupil_petals")
pmap = np.where(ref["petal_map"] < 0, np.nan, ref["petal_map"]).astype(float)
ax.imshow(pmap, cmap="Pastel1", origin="upper")
cy, cx = (pmap.shape[0] - 1) / 2, (pmap.shape[1] - 1) / 2
for k, ang in enumerate(ref["angle_deg"]):
    r = 0.33 * pmap.shape[0]
    ax.text(cx + r * np.cos(np.radians(ang)), cy - r * np.sin(np.radians(ang)),
            "ref" if k == ref["ref"] else f"p{k}", ha="center", va="center", weight="bold")
ax.axis("off")
save(fig, "pupil_petals")
print("done")
