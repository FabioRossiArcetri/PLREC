"""Analysis and plots of piston2.ipynb: why is piston 2 the easiest piston?

See ml_reconstruct.petals for the twin symmetry, and scripts/piston2_twins.py /
scripts/piston2_reref.py for the simulations and trainings these read.
"""
from __future__ import annotations

import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from .data import rmse_nm
from .unwrap import wrap


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def plot_pupil_and_lantern(geom, core_xy):
    """The pupil petals (labelled by piston mode; R = reference), the opposite
    petal pairs, and the angles of the lantern's first ring of output cores."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    pm = np.ma.masked_less(geom["petal_map"], 0)
    ax[0].imshow(pm, cmap="tab10", vmin=0, vmax=9, origin="upper")
    cy, cx = (np.array(pm.shape) - 1) / 2
    for k, a in enumerate(geom["angle_deg"]):
        r = 0.33 * pm.shape[0]
        x, y = cx + r * np.cos(np.radians(a)), cy - r * np.sin(np.radians(a))
        lab = "R" if k == geom["ref"] else f"p{k}"
        ax[0].text(x, y, lab, ha="center", va="center", fontsize=14, weight="bold", color="w")
    for k, o in enumerate(geom["opposite"]):
        if k < o:
            a = np.radians(geom["angle_deg"][k])
            ax[0].plot([cx + 0.45 * cx * np.cos(a), cx - 0.45 * cx * np.cos(a)],
                       [cy - 0.45 * cy * np.sin(a), cy + 0.45 * cy * np.sin(a)], "w:", lw=1)
    ax[0].set_title("pupil petals (R: reference); dotted: opposite pairs"); ax[0].axis("off")
    xy = np.asarray(core_xy, float)
    ax[1].scatter(xy[:, 0], xy[:, 1], s=300, c="lightgray", edgecolors="k")
    for i, (x, y) in enumerate(xy):
        ax[1].text(x, y, str(i), ha="center", va="center", fontsize=7)
    rmax = np.abs(xy).max() * 1.25
    for k, a in enumerate(geom["angle_deg"]):
        ax[1].plot([0, rmax * np.cos(np.radians(a))], [0, rmax * np.sin(np.radians(a))], "--", lw=1,
                   color=plt.cm.tab10(k))
        ax[1].text(1.08 * rmax * np.cos(np.radians(a)), 1.08 * rmax * np.sin(np.radians(a)),
                   "R" if k == geom["ref"] else f"p{k}", ha="center", va="center")
    ax[1].set_aspect("equal"); ax[1].set_title("lantern output cores, with the petal directions")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# per-piston errors
# --------------------------------------------------------------------------- #
def piston_error_table(estimates):
    """{name: (estimates, truth)} -> per-piston RMSE and median |error| table."""
    import pandas as pd
    rows = {}
    for name, (P, C) in estimates.items():
        e = np.abs(P - C)
        rows[name] = dict(**{f"RMSE p{k}": v for k, v in enumerate(rmse_nm(P, C))},
                          **{f"median p{k}": v for k, v in enumerate(np.median(e, 0))})
    return pd.DataFrame(rows).T


def odd_part(C, T, period):
    """|p + T p| / 2 per configuration, modulo the period: 0 for a
    point-symmetric (even) pupil phase, whose twin is its sign flip -p."""
    return np.linalg.norm(wrap(C + C @ T.T, period), axis=1) / 2


def errors_by_odd_part(P, C, T, period, quantiles=(0, 2, 5, 10, 25, 50, 100), gross_nm=50.0):
    """RMSE per piston and share of gross errors in bins of the odd part
    (quantiles of the configurations)."""
    import pandas as pd
    odd = odd_part(C, T, period)
    out = np.abs(P - C).max(1) > gross_nm
    edges = np.percentile(odd, quantiles)
    rows = []
    for lo, hi, qa, qb in zip(edges[:-1], edges[1:], quantiles[:-1], quantiles[1:]):
        m = (odd >= lo) & (odd <= hi)
        rows.append(dict(bin=f"{qa}-{qb}%", odd_nm=f"{lo:.0f}-{hi:.0f}", n=int(m.sum()),
                         **{f"p{k}": v for k, v in enumerate(rmse_nm(P[m], C[m]))},
                         **{f"> {gross_nm:g} nm %": 100 * out[m].mean()}))
    return pd.DataFrame(rows)


def plot_errors_vs_odd_part(P, C, T, period, bins=30):
    """|error| of each piston against the odd part of the configuration: the
    median and the 99th percentile in bins."""
    odd = odd_part(C, T, period)
    edges = np.percentile(odd, np.linspace(0, 100, bins + 1))
    mid = 0.5 * (edges[1:] + edges[:-1])
    fig, ax = plt.subplots(1, 2, figsize=(13, 4), sharex=True)
    for k in range(C.shape[1]):
        e = np.abs(P[:, k] - C[:, k])
        stats = np.array([[np.median(e[(odd >= a) & (odd <= b)]), np.percentile(e[(odd >= a) & (odd <= b)], 99)]
                          for a, b in zip(edges[:-1], edges[1:])])
        kw = dict(lw=2.5 if k == 2 else 1, label=f"piston {k}")
        ax[0].plot(mid, stats[:, 0], "o-", ms=3, **kw); ax[1].plot(mid, stats[:, 1], "o-", ms=3, **kw)
    for a, t in zip(ax, ("median |error|", "99th percentile of |error|")):
        a.set_yscale("log"); a.set_xlabel("odd part |p + Tp| / 2 (nm)"); a.set_title(t); a.grid(alpha=0.3, which="both")
    ax[0].set_ylabel("nm"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# twin separability (scripts/piston2_twins.py)
# --------------------------------------------------------------------------- #
def twin_ratios(signals, n_base):
    """|s(p) - s(Tp)| / |s(p) - s(p + D)| per base configuration: how far its
    twin is from it, relative to a configuration at the same distance (D)."""
    S0, ST, SD = signals[:n_base], signals[n_base:2 * n_base], signals[2 * n_base:]
    return np.linalg.norm(S0 - ST, axis=1) / np.linalg.norm(S0 - SD, axis=1)


def load_twin_ratios(directory, family=""):
    """{wavelength_nm: ratios} of the channel-power files, and the 800 nm
    camera ratios (None if missing). family: "" (random) or "_even"."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, f"channels{family}_[0-9]*.npz"))):
        with np.load(path) as f:
            out[float(f["wavelength_nm"])] = twin_ratios(f["signals"], int(f["n_base"]))
    cam = os.path.join(directory, f"camera{family}_800nm.npz")
    camera = None
    if os.path.exists(cam):
        with np.load(cam) as f:
            camera = twin_ratios(f["signals"], int(f["n_base"]))
    return out, camera


def plot_twin_ratios(families):
    """Median and quartiles of the twin ratio vs. wavelength, one line per
    family {label: (ratios by wavelength, camera ratios)}; 1 = the twin is as
    distinguishable as any configuration that far away, 0 = indistinguishable."""
    plt.figure(figsize=(9, 4))
    for i, (label, (by_wl, cam)) in enumerate(families.items()):
        if not by_wl:
            continue
        wl = np.array(sorted(by_wl)); q = np.array([np.percentile(by_wl[w], [25, 50, 75]) for w in wl])
        c = plt.cm.tab10(i)
        plt.fill_between(wl, q[:, 0], q[:, 2], color=c, alpha=0.2)
        plt.plot(wl, q[:, 1], "o-", ms=3, color=c, label=f"{label}: channel powers (median, quartiles)")
        if cam is not None:
            plt.errorbar([800], [np.median(cam)], yerr=[[np.median(cam) - np.percentile(cam, 25)],
                         [np.percentile(cam, 75) - np.median(cam)]], fmt="s", color=c, ms=8, capsize=4,
                         label=f"{label}: camera at 800 nm")
    plt.axhline(1, color="k", lw=0.8, ls=":")
    plt.ylim(bottom=0); plt.xlabel("wavelength (nm)")
    plt.ylabel("|s(p) - s(Tp)| / |s(p) - s(p+D)|")
    plt.title("how distinguishable a configuration is from its twin"); plt.legend(fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def perwl_piston_ratio(runs):
    """Per wavelength: wrapped-piston RMSE of the other pistons over that of
    piston 2, for each per-wavelength MLP architecture (workbench Part C)."""
    import pandas as pd
    g = runs[runs.role == "perwl"]
    rows = []
    for (d, w, lam), r in g.groupby(["depth", "width", "wavelength_nm"]):
        v = np.array(r.val_rmse.iloc[0])
        rows.append(dict(arch=f"{d}x{w}", wavelength_nm=lam, ratio=np.mean(v[[0, 1, 3, 4]]) / v[2]))
    return pd.DataFrame(rows)


def plot_ratio_vs_twin(ratio_df, twin_by_wl):
    """Piston-2 advantage of the per-wavelength MLPs vs. wavelength, next to the
    median twin ratio of the simulation."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for arch, g in ratio_df.groupby("arch"):
        ax.plot(g.wavelength_nm, g.ratio, "o-", ms=3, label=f"per-wl MLP {arch}")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.set_xlabel("wavelength (nm)"); ax.set_ylabel("RMSE(p0,p1,p3,p4) / RMSE(p2)")
    ax.legend(fontsize=7, loc="upper left"); ax.grid(alpha=0.3)
    if twin_by_wl:
        ax2 = ax.twinx()
        wl = np.array(sorted(twin_by_wl))
        ax2.plot(wl, [np.median(twin_by_wl[w]) for w in wl], "k--", lw=2, label="median twin ratio (sim.)")
        ax2.set_ylabel("median twin ratio"); ax2.legend(fontsize=7, loc="upper right")
    ax.set_title("piston-2 advantage of the per-wavelength models vs. twin separability")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# re-referencing (scripts/piston2_reref.py)
# --------------------------------------------------------------------------- #
def reref_table(directory, opposite, n_petals=6):
    """RMSE of each petal's piston for each choice of the reference petal:
    rows reference petal, columns petal (NaN on the reference itself), plus
    the easiest petal and the petal opposite the reference."""
    import pandas as pd
    rows = {}
    for r in range(n_petals):
        path = os.path.join(directory, f"reref_{r}.npz")
        if not os.path.exists(path):
            continue
        with np.load(path) as f:
            rm = rmse_nm(f["pred_va"], f["true_va"]); ids = list(f["petal_ids"])
        row = {f"petal {k}": (rm[ids.index(k)] if k in ids else np.nan) for k in range(n_petals)}
        row["easiest"] = ids[int(np.argmin(rm))]
        row["opposite of ref"] = int(opposite[r])
        rows[r] = row
    return pd.DataFrame(rows).T.rename_axis("reference petal")


def plot_reref(table, n_petals=6):
    """Heat map of reref_table: the easiest petal of each row framed in white,
    the petal opposite the reference marked with a dot."""
    v = table[[f"petal {k}" for k in range(n_petals)]].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(v, cmap="viridis_r")
    for i, (_, r) in enumerate(table.iterrows()):
        ax.add_patch(plt.Rectangle((int(r["easiest"]) - 0.5, i - 0.5), 1, 1, fill=False, ec="w", lw=2.5))
        ax.plot(int(r["opposite of ref"]), i, "w.", ms=8)
        for k in range(n_petals):
            if np.isfinite(v[i, k]):
                ax.text(k, i, f"{v[i, k]:.1f}", ha="center", va="center", fontsize=8, color="k")
    ax.set_xticks(range(n_petals), [f"petal {k}" for k in range(n_petals)])
    ax.set_yticks(range(len(table)), [f"ref {int(r)}" for r in table.index])
    ax.set_title("RMSE (nm) by reference petal\n(framed: easiest; dot: opposite the reference)")
    plt.colorbar(im, ax=ax); plt.tight_layout(); plt.show()


def twin_combinations(T, opposite, ref):
    """{name: unit vector a} of piston combinations a @ p: the differences across
    the opposite petal pairs (twin-invariant, a @ T = a) and the twin-odd
    combinations (a @ T = -a), for pistons p of the petals other than ref."""
    keep = [k for k in range(len(opposite)) if k != ref]
    e = lambda k: np.eye(len(keep))[keep.index(k)]
    combos, odd = {}, {}
    for k in keep:
        o = opposite[k]
        if o == ref:                       # the reference's partner: its piston itself
            combos[f"p{k} - p_ref"] = e(k)
        elif k < o:                        # difference (invariant) and sum (odd) of the pair
            combos[f"p{k} - p{o}"] = (e(k) - e(o)) / np.sqrt(2)
            a = e(k) + e(o) - e(opposite[ref])         # a @ T p = -(a @ p)
            odd[f"p{k} + p{o} - p{opposite[ref]}"] = a / np.linalg.norm(a)
    combos.update(odd)
    for name, a in combos.items():
        assert np.allclose(a @ T, a) or np.allclose(a @ T, -a), name
    return combos


def combination_errors(estimates, combos):
    """RMSE, median and 99th percentile of |error| of each unit-norm piston
    combination, for {name: (estimates, truth)}."""
    import pandas as pd
    rows = []
    for name, (P, C) in estimates.items():
        for cname, a in combos.items():
            e = (P - C) @ a
            rows.append(dict(model=name, combination=cname, rmse=np.sqrt(np.mean(e ** 2)),
                             median=np.median(np.abs(e)), p99=np.percentile(np.abs(e), 99)))
    return pd.DataFrame(rows).pivot(index="combination", columns="model", values=["rmse", "median", "p99"])


# --------------------------------------------------------------------------- #
# folds along lines through the flat wavefront (scripts/piston2_twins.py --family lines)
# --------------------------------------------------------------------------- #
def load_lines(directory):
    """{label: npz contents} of the line files: 'camera 800 nm' and 'channels <nm> nm'."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*_lines_*.npz"))):
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
        out["camera 800 nm" if "camera" in path else f"channels {float(d['wavelength_nm']):.0f} nm"] = d
    return out


def fold_asymmetry(d):
    """{kind: (|t| values, (n_lines, n_t) asymmetry)} with the asymmetry
    |s(t v) - s(-t v)| / |s(t v) - s(0)| along each line: 0 for a perfect fold
    (the sign of t invisible), ~1 or more when +t and -t look different."""
    out = {}
    for kind in np.unique(d["kind"]):
        A, ts = [], None
        for j in np.unique(d["line"][d["kind"] == kind]):
            m = (d["kind"] == kind) & (d["line"] == j)
            t, S = d["t"][m], d["signals"][m]
            o = np.argsort(t); t, S = t[o], S[o]
            n = len(t) // 2                                  # t = 0 in the middle
            pos, neg = S[n + 1:], S[:n][::-1]
            A.append(np.linalg.norm(pos - neg, axis=1) / np.linalg.norm(pos - S[n], axis=1))
            ts = t[n + 1:]
        out[str(kind)] = (ts, np.array(A))
    return out


def plot_folds(lines):
    """Median (and quartiles) fold asymmetry vs |t| for each kind of line, one
    panel per measurement (camera, channel powers per wavelength)."""
    fig, ax = plt.subplots(1, len(lines), figsize=(4.2 * len(lines), 3.6), sharey=True, squeeze=False)
    for a, (label, d) in zip(ax[0], lines.items()):
        for i, (kind, (t, A)) in enumerate(fold_asymmetry(d).items()):
            q = np.percentile(A, [25, 50, 75], axis=0)
            a.fill_between(300 * t, q[0], q[2], color=plt.cm.tab10(i), alpha=0.2)
            a.plot(300 * t, q[1], "o-", ms=3, color=plt.cm.tab10(i), label=kind)
        a.set_yscale("log"); a.grid(alpha=0.3, which="both"); a.set_title(label, fontsize=10)
        a.set_xlabel("largest piston along the line (nm)")
    ax[0, 0].set_ylabel("|s(tv) - s(-tv)| / |s(tv) - s(0)|"); ax[0, -1].legend(fontsize=8)
    plt.suptitle("is the sign along a line visible? (low = folded, sign invisible)", y=1.02)
    plt.tight_layout(); plt.show()


def plot_line_signals(d, kinds=("point-symmetric", "twin-invariant"), line=0):
    """The core signals along one line of each kind, against the largest piston."""
    fig, ax = plt.subplots(1, len(kinds), figsize=(6 * len(kinds), 3.6), squeeze=False)
    for a, kind in zip(ax[0], kinds):
        m = (d["kind"] == kind) & (d["line"] == line)
        o = np.argsort(d["t"][m]); t, S = d["t"][m][o], d["signals"][m][o]
        for c in range(S.shape[1]):
            a.plot(300 * t, S[:, c], lw=1, color=plt.cm.turbo(c / (S.shape[1] - 1)))
        a.set_title(f"{kind} line {line}: the {S.shape[1]} core signals", fontsize=10)
        a.set_xlabel("largest piston along the line (nm)"); a.grid(alpha=0.3)
    ax[0, 0].set_ylabel("signal"); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# the twin in the focal plane
# --------------------------------------------------------------------------- #
def complex_image(E, decades=4.0):
    """RGB "domain colouring" of a complex field: hue = phase (cyclic HSV),
    brightness = intensity on a log scale over the top ``decades`` of the peak.
    No masking is needed: where the field is weak (dark rings, phase
    singularities) the phase simply fades to black."""
    from matplotlib.colors import hsv_to_rgb
    I = np.abs(E) ** 2
    v = np.clip(1 + np.log10(I / I.max() + 1e-30) / decades, 0, 1)
    h = (np.angle(E) + np.pi) / (2 * np.pi)
    return hsv_to_rgb(np.stack([h, np.ones_like(h), v], axis=-1))


def phase_colorbar(ax):
    """Colour bar of complex_image's phase hue."""
    sm = plt.cm.ScalarMappable(cmap="hsv", norm=plt.Normalize(-np.pi, np.pi))
    cb = plt.colorbar(sm, ax=ax, fraction=0.046, ticks=[-np.pi, 0, np.pi])
    cb.ax.set_yticklabels(["-pi", "0", "pi"]); cb.set_label("phase (hue)")


def focal_fields(ifunc, p, C, half_width_um=14.0):
    """Focal-plane fields of the piston configurations C (n, 5) [nm], as they
    reach the lantern (batch_pipeline's pupil -> zero-padded FFT), cropped to
    +-half_width_um. Returns ``(E (n, m, m) complex, extent [um])``."""
    from .simulate import bp
    fg = bp.IncidentFieldGenerator(p, ifunc)
    coeff = np.zeros((len(C), fg.num_modes)); coeff[:, :C.shape[1]] = C
    E = fg.apply_ef_to_lantern(fg.generate_field_profiles_batch(coeff))
    c, h = E.shape[1] // 2, int(round(half_width_um / p["pixel_scale_um"]))
    ext = np.array([-h - 0.5, h + 0.5, -h - 0.5, h + 0.5]) * p["pixel_scale_um"]
    return E[:, c - h:c + h + 1, c - h:c + h + 1], ext


def pupil_phase(geom, p, wavelength_nm):
    """Pupil phase map [rad, wrapped] of the pistons p (reference petal at 0)."""
    piston = np.append(np.asarray(p, float), 0.0)                 # petal order, reference last
    ph = np.where(geom["petal_map"] >= 0, piston[np.clip(geom["petal_map"], 0, None)], np.nan)
    return np.angle(np.exp(2j * np.pi * ph / wavelength_nm))


def plot_twin_psfs(geom, configs, E, ext, wavelength_nm, r_clad_um=9.0):
    """One row per configuration {name: pistons}: pupil phase, focal-plane
    intensity (log) and focal-plane field (phase as hue, intensity as
    brightness: complex_image), with the lantern's input cladding boundary."""
    names = list(configs)
    fig, ax = plt.subplots(len(names), 3, figsize=(11, 3.4 * len(names)))
    I_max = np.max(np.abs(E) ** 2)
    for i, name in enumerate(names):
        I, ph = np.abs(E[i]) ** 2, np.angle(E[i])
        ax[i, 0].imshow(pupil_phase(geom, configs[name], wavelength_nm), cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax[i, 0].set_title(f"{name}\npistons {np.round(configs[name]).astype(int)} nm", fontsize=8)
        ax[i, 1].imshow(np.log10(I / I_max + 1e-6), extent=ext, origin="lower", cmap="inferno", vmin=-4, vmax=0)
        ax[i, 2].imshow(complex_image(E[i]), extent=ext, origin="lower")
        phase_colorbar(ax[i, 2])
        for a in ax[i, 1:]:
            a.add_patch(plt.Circle((0, 0), r_clad_um, fill=False, ec="c", lw=1, ls="--"))
            a.set_xlabel("um")
        ax[i, 0].axis("off")
    ax[0, 1].set_title("focal-plane intensity (log10)\ndashed: lantern input", fontsize=9)
    ax[0, 2].set_title("focal-plane field: hue = phase,\nbrightness = intensity (4 decades)", fontsize=9)
    plt.tight_layout(); plt.show()


def plot_example_signals(d, groups):
    """Bar chart of the lantern's camera signals of each group of example
    configurations (names of the camera_examples file), one panel per group."""
    names = list(d["names"])
    fig, ax = plt.subplots(1, len(groups), figsize=(7 * len(groups), 3.6), squeeze=False)
    for a, (title, members) in zip(ax[0], groups.items()):
        w = 0.8 / len(members)
        for j, n in enumerate(members):
            a.bar(np.arange(19) + (j - len(members) / 2 + 0.5) * w, d["signals"][names.index(n)], w, label=n)
        a.set_xlabel("output core"); a.set_xticks(range(19)); a.set_title(title, fontsize=10)
        a.legend(fontsize=8); a.grid(alpha=0.3, axis="y")
    ax[0, 0].set_ylabel("camera signal")
    plt.tight_layout(); plt.show()


def signal_distance_table(d, ref_signals):
    """|s(a) - s(b)| between the example configurations, in units of the typical
    distance between two random training configurations."""
    import pandas as pd
    names, S = list(d["names"]), d["signals"]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(ref_signals), (2, 2000))
    scale = np.median(np.linalg.norm(ref_signals[i] - ref_signals[j], axis=1))
    D = np.linalg.norm(S[:, None] - S[None], axis=2) / scale
    return pd.DataFrame(D, index=names, columns=names)
