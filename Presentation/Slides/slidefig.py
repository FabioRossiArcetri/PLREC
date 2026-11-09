#!/usr/bin/env python
"""Slide figures at their size on the slide, with one font size for all.

Every figure of the slides is a pickled matplotlib figure:
  - sim figures: make_figures.py / make_ambiguity_figure.py   -> figures/pkl/<name>.pkl
  - notebook figures: pickled while the notebooks execute (run_lantern.sh,
    leonardo/extras.slurm; PLREC/tools/ipython)              -> PLREC/runs/<run>/figs/
This module redraws each one at the size of its slot (SLOTS, cm) with all
text at FONT_PT, and writes figures/slides/<name>.pdf; main_v2.tex includes
them at natural size, so the text has the same size on every slide.

    python slidefig.py              # every slot whose pickle exists
    python slidefig.py NAME ...     # some slots
    python slidefig.py --list       # slots and their sources
    python slidefig.py --fallback   # slots without a pickle get the old PNG fitted (interim)
"""
from __future__ import annotations

import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.collections import Collection, PathCollection
from matplotlib.colors import to_rgba
from matplotlib.ticker import AutoMinorLocator, LogLocator, MaxNLocator   # noqa: E402
from matplotlib.legend import Legend                # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Patch                # noqa: E402
from matplotlib.text import Text                    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLREC = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))
PKL = os.path.join(HERE, "figures", "pkl")
OUT = os.path.join(HERE, "figures", "slides")
FONT_PT = 5.6
CM = 1 / 2.54
FALLBACK = False
# for figures drawn directly at slide size (make_figures.py, make_ambiguity_figure.py)
RC = {"font.size": FONT_PT, "axes.titlesize": FONT_PT, "axes.labelsize": FONT_PT,
      "xtick.labelsize": FONT_PT, "ytick.labelsize": FONT_PT, "legend.fontsize": FONT_PT,
      "figure.titlesize": FONT_PT, "lines.linewidth": 1.0, "lines.markersize": 3.5,
      "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
      "xtick.major.size": 2.5, "ytick.major.size": 2.5, "axes.titlepad": 3.0,
      "legend.handlelength": 1.5, "figure.dpi": 150,
      "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02}

# --------------------------------------------------------------------------- #
# the slots: name -> (source, width cm, height cm[, options])
#   source ("sim", name)                          figures/pkl/<name>.pkl
#   source ("nb", run, notebook, cell, k)         the k-th figure of cell <cell>
#          (index in the notebook, markdown included) of runs/<run>/<notebook>.ipynb
# options: legend="below" | "right" | None, ncol=<legend columns>, title=False
# The slide is 16 x 9 cm; the text width is ~14.5 cm, a frame's body ~6.5 cm tall.
# --------------------------------------------------------------------------- #
W_FULL, W_HALF, W_WIDE, W_NARROW = 14.0, 7.0, 8.6, 6.0
# the 5-panel unwrap scatter: "piston k" titles, one x / y label, few ticks
SCATTER = dict(retitle=[(r"^.*?piston (\d).*$", r"piston \1")], share=True, nbins=3)


def _sim(lantern, fig, w, h, **o):
    return f"{lantern}_{fig}", (("sim", f"{lantern}_{fig}"), w, h, o)


# --------------------------------------------------------------------------- #
# one colour per reconstructor, in every figure
#
# The notebooks plot with matplotlib's default cycle, so a figure's colour depends on
# the order in which the methods happen to be drawn: the bar charts sort by accuracy,
# the wavefront-error plot uses a fixed order, and the same method ends up with a
# different colour on neighbouring slides. The figures are re-drawn here, so the colour
# is set here too, keyed on the method and not on the drawing order.
# --------------------------------------------------------------------------- #
CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]      # tab10
PALETTE = {"linear": CYCLE[0], "symbolic": CYCLE[1], "dense d2": CYCLE[2],
           "dense d3": CYCLE[3], "omp": CYCLE[4], "closed form": CYCLE[5],
           "mlp": CYCLE[6]}
# degrees of freedom per piston, to recognise a method in a figure that carries no legend
DOF = {
    "pl19_1200": {"single": {"linear": 40, "symbolic": 58, "dense d2": 420, "dense d3": 3080,
                             "omp": 2562, "closed form": 124, "mlp": 7375},
                  "spectral": {"linear": 840, "dense d2": 8820, "omp": 53886,
                               "dense d3": 64680, "mlp": 154871, "closed form": 2474}},
}


def method_key(label):
    """'sparse poly (OMP), K=1282 (2,562 dof/piston)' -> 'omp'"""
    t = (label or "").lower()
    if not t or t.startswith("_"):
        return None
    for key in ("mlp", "closed form", "closed-form", "omp", "sparse",
                "symbolic", "degree 3", "degree 2", "linear"):
        if key in t:
            return {"sparse": "omp", "degree 2": "dense d2", "degree 3": "dense d3"}.get(key, key)
    return None


def _rgb(value):
    """('black' | 'none' | (r, g, b) | array) -> rounded (r, g, b) tuple, or None."""
    try:
        arr = np.atleast_2d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        try:
            arr = np.atleast_2d(np.asarray(to_rgba(value), dtype=float))
        except (TypeError, ValueError):
            return None
    if arr.size < 3:
        return None
    return tuple(round(float(x), 4) for x in arr[0][:3])


def _colour_of(handle):
    for getter in ("get_color", "get_facecolor", "get_edgecolor"):
        try:
            c = getattr(handle, getter)()
        except Exception:
            continue
        rgb = _rgb(c)
        if rgb is not None:
            return rgb
    return None


def colour_map(fig):
    """{old colour: canonical colour} from the legends of the figure."""
    cmap = {}
    legs = list(fig.legends)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legs.append(ax.get_legend())
    for leg in legs:
        for text, handle in zip(leg.get_texts(), leg.legend_handles):
            key = method_key(text.get_text())
            old = _colour_of(handle)
            if key and old is not None:
                cmap[old] = PALETTE[key]
    return cmap


def dof_recolour(fig, table):
    """Colour the points of a legend-less dof plot by their abscissa.

    Each point is placed at the degrees of freedom of its method, which are far apart,
    so the position identifies the method without any help from the current colour.
    Returns the number of points recoloured.
    """
    n = 0
    for ax in fig.axes:
        for art in list(ax.lines) + list(ax.collections):
            xy = _xy_of(art)
            if xy is None:
                continue
            xs, _ = xy
            if not xs.size:
                continue
            x = float(np.nanmedian(xs))
            key = min(table, key=lambda k: abs(table[k] - x) / max(1.0, table[k]))
            if abs(table[key] - x) / max(1.0, table[key]) > 0.10:
                continue
            colour = PALETTE[key]
            try:
                art.set_color(colour)
            except Exception:
                pass
            for setter in ("set_markerfacecolor", "set_markeredgecolor"):
                try:
                    getattr(art, setter)(colour)
                except Exception:
                    pass
            for setter in ("set_facecolor", "set_edgecolor"):
                try:
                    getattr(art, setter)(colour)
                except Exception:
                    pass
            n += 1
    return n


def dof_colour_map(fig, table):
    """{old colour: canonical colour} for a plot with no legend.

    The dof table is the key: the points are far apart on the abscissa, so each one
    identifies its method exactly. Text annotations label the points but sit at the
    same coordinates, so they are only used for methods the table does not know.
    """
    cmap = {}
    for ax in fig.axes:
        arts = [a for a in list(ax.lines) + list(ax.collections) if _xy_of(a) is not None]
        rest = []
        for art in arts:
            xs, _ = _xy_of(art)
            if not xs.size:
                continue
            x = float(np.nanmedian(xs))
            key = min(table, key=lambda k: abs(table[k] - x) / max(1.0, table[k]))
            if abs(table[key] - x) / max(1.0, table[key]) < 0.10:      # the same point
                old = _colour_of(art)
                if old is not None:
                    cmap[old] = PALETTE[key]
            else:
                rest.append((art, x))
        # anything the table does not place: take the label nearest in x
        for art, x in rest:
            for txt in ax.texts:
                key = method_key(txt.get_text())
                if key is None:
                    continue
                tx = txt.get_position()[0]
                if abs(np.log10(max(tx, 1e-9)) - np.log10(max(x, 1e-9))) < 0.15:
                    old = _colour_of(art)
                    if old is not None:
                        cmap[old] = PALETTE[key]
                    break
    return cmap


def _xy_of(art):
    try:
        data = art.get_offsets() if isinstance(art, PathCollection) else None
        if data is not None and len(data):
            return np.asarray(data)[:, 0], np.asarray(data)[:, 1]
        x, y = art.get_data()
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    except Exception:
        return None


def apply_colours(fig, cmap):
    """Recolour every artist, and the legend handles, through *cmap*."""
    if not cmap:
        return 0
    n = 0
    for art in fig.findobj(Line2D):
        for name in ("color", "markerfacecolor", "markeredgecolor"):
            try:
                cur = art.get_color() if name == "color" else getattr(art, "get_" + name)()
            except Exception:
                continue
            key = _rgb(cur)
            if key is not None and key in cmap:
                if name == "color":
                    art.set_color(cmap[key])
                else:
                    getattr(art, "set_" + name)(cmap[key])
                n += 1
    for art in fig.findobj(Patch):
        for name in ("facecolor", "edgecolor"):
            key = _rgb(getattr(art, "get_" + name)())
            if key is not None and key in cmap:
                getattr(art, "set_" + name)(cmap[key])
                n += 1
    for art in fig.findobj(Collection):
        for name in ("facecolor", "edgecolor"):
            try:
                cols = getattr(art, "get_" + name)()
            except Exception:
                continue
            cols = np.atleast_2d(np.asarray(cols, dtype=float))
            new = np.array(cols)
            hit = False
            for i, row in enumerate(cols):
                key = _rgb(row)
                if key is not None and key in cmap:
                    new[i, :3] = cmap[key]
                    hit = True
            if hit:
                getattr(art, "set_" + name)(new)
                n += 1
    legs = list(fig.legends)
    for ax in fig.axes:
        if ax.get_legend() is not None:
            legs.append(ax.get_legend())
    for leg in legs:
        for handle in leg.legend_handles:
            for name in ("color", "facecolor", "edgecolor"):
                try:
                    key = _rgb(getattr(handle, "get_" + name)())
                except Exception:
                    continue
                if key is not None and key in cmap:
                    try:
                        getattr(handle, "set_" + name)(cmap[key])
                        n += 1
                    except Exception:
                        pass
    return n


def symbolic_colours(fig):
    """Symbolic regression: its own scheme, the best search of each class set in orange.

    The curves are the Operon Pareto fronts of the four piston classes plus two reference
    lines. The class with the lowest held-out error keeps orange and a heavier line; the
    others go muted grey-blue/green/purple, the references grey.
    """
    muted = ["#4c72b0", "#55a868", "#8172b2", "#937860"]
    ax = fig.axes[0]
    curves = [ln for ln in ax.lines
              if (ln.get_label() or "").lower().startswith(("operon", "pysr"))]
    if not curves:
        return 0
    best = min(curves, key=lambda ln: np.nanmin(np.asarray(ln.get_ydata(), float)))
    i = 0
    for ln in curves:
        if ln is best:
            ln.set_color("#ff7f0e")
            ln.set_linewidth(max(ln.get_linewidth(), 1.4))
            ln.set_zorder(6)
        else:
            ln.set_color(muted[i % len(muted)])
            ln.set_zorder(4)
            i += 1
    for ln in ax.lines:
        if (ln.get_label() or "").lower() in ("linear", "dense poly, degree 2", "no skill"):
            ln.set_color("#7a7a7a")
    handles = curves + [ln for ln in ax.lines
                        if (ln.get_label() or "").lower() in ("linear", "dense poly, degree 2")]
    labels = [ln.get_label() for ln in handles]
    old = ax.get_legend()
    ncol = getattr(old, "_ncols", 1) if old is not None else 1
    ax.legend(handles=handles, labels=labels, ncol=ncol, fontsize=FONT_PT)
    return len(curves)


def place_labels(fig, spec):
    """Put each method label at its own point plus an offset in axes fractions.

    spec: {label text: (dx, dy, ha, va)}, offsets relative to the point the label belongs
    to (the nearest one in log-x). These labels are drawn in axes coordinates, so the
    offsets live there too.
    """
    n = 0
    for ax in fig.axes:
        texts = [t for t in ax.texts if (t.get_text() or "").strip()]
        if not texts:
            continue
        pts = []
        for art in list(ax.lines) + list(ax.collections):
            xy = _xy_of(art)
            if xy is None:
                continue
            for x, y in zip(np.atleast_1d(xy[0]), np.atleast_1d(xy[1])):
                if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                    pts.append((float(x), float(y)))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if not pts or xlim[0] <= 0 or ylim[0] <= 0:
            continue
        lx0, lx1 = np.log10(xlim[0]), np.log10(xlim[1])
        ly0, ly1 = np.log10(ylim[0]), np.log10(ylim[1])
        for t in texts:
            row = spec.get(t.get_text().strip())
            if row is None:
                continue
            dx, dy, ha, va = row
            tx, ty = t.get_position()
            px, py = min(pts, key=lambda q: abs(np.log10(q[0]) - np.log10(max(tx, 1e-9))))
            fx = (np.log10(px) - lx0) / (lx1 - lx0)
            fy = (np.log10(py) - ly0) / (ly1 - ly0)
            t.set_position((float(np.clip(fx + dx, 0.02, 0.98)),
                            float(np.clip(fy + dy, 0.03, 0.97))))
            t.set_ha(ha)
            t.set_va(va)
            n += 1
    return n


def tidy_labels(fig, scale=0.80):
    """Smaller method labels, anchored to their point and pushed outwards.

    The plots that label points with text put the text at the point itself, so it sits on
    the marker and, for the points on the right, overruns the frame. Each label is
    re-anchored to its nearest point, offset up or down depending on which half of the
    axes it is in, and kept inside the axes.
    """
    n = 0
    try:
        fig.canvas.draw()            # the axes limits must be real before clamping
    except Exception:
        pass
    for ax in fig.axes:
        texts = [t for t in ax.texts if (t.get_text() or "").strip()]
        if not texts:
            continue
        pts = []
        for art in list(ax.lines) + list(ax.collections):
            xy = _xy_of(art)
            if xy is None:
                continue
            for x, y in zip(np.atleast_1d(xy[0]), np.atleast_1d(xy[1])):
                if np.isfinite(x) and np.isfinite(y) and x > 0 and y > 0:
                    pts.append((float(x), float(y)))
        if not pts:
            continue
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        if xlim[0] <= 0 or ylim[0] <= 0:          # not a log-log plot after all
            continue
        mid_x = np.sqrt(xlim[0] * xlim[1])
        mid_y = np.sqrt(ylim[0] * ylim[1])
        for t in texts:
            t.set_fontsize(max(3.2, t.get_fontsize() * scale))
            tx, ty = t.get_position()
            # these labels are placed in axes fractions, not in data coordinates
            if t.get_transform() is ax.transData:
                continue
            right = tx > 0.5
            dx = -0.015 if right else 0.015
            dy = 0.035 if ty < 0.5 else -0.035
            t.set_position((float(np.clip(tx + dx, 0.02, 0.98)),
                            float(np.clip(ty + dy, 0.03, 0.97))))
            t.set_ha("right" if right else "left")
            t.set_va("bottom" if dy > 0 else "top")
            n += 1
    return n


def rebuild_legend(fig):
    """Rebuild a bar chart's legend from its containers.

    The notebooks sort the legend entries by accuracy but pass the handles in drawing
    order, so the pickled legend pairs a label with another method's colour. The
    containers carry label and colour together, so the legend is rebuilt from them.
    """
    n = 0
    for ax in fig.axes:
        conts = [c for c in ax.containers if (c.get_label() or "").strip("_")]
        if len(conts) < 2:
            continue
        old = ax.get_legend()
        if old is None:
            continue
        kw = {"title": old.get_title().get_text() or None, "loc": "best", "frameon": old.get_frame_on()}
        ncol = getattr(old, "_ncols", 1) or 1
        ax.legend(handles=conts, labels=[c.get_label() for c in conts],
                  ncol=ncol, fontsize=FONT_PT, **{k: v for k, v in kw.items() if v is not None})
        n += 1
    return n


def canonical_colours(name, fig):
    """Set the one-colour-per-method palette on a figure loaded from a pickle."""
    if "sr_fronts" in name or "sr_scatter" in name:
        return symbolic_colours(fig)
    cmap = colour_map(fig)
    if cmap:
        n = apply_colours(fig, cmap)
        return n + rebuild_legend(fig)
    # no legend: the points are placed at the dofs of their method
    src = SLOTS[name][0]
    kind = "spectral" if "spectral" in name else "single"
    table = DOF.get(src[1], {}).get(kind, {}) if src[0] == "nb" else {}
    if table:
        return dof_recolour(fig, table)
    return 0


def _nb(lantern, fig, notebook, cell, k, w, h, run=None, **o):
    return f"{lantern}_{fig}", (("nb", run or lantern, notebook, cell, k), w, h, o)


SLOTS = dict([
    # part 1: the lantern (reference design)
    _sim("pl19_800", "modes_input", W_HALF, 2.6),
    _sim("pl19_800", "geometry", W_WIDE, 4.8),
    _sim("pl19_800", "neff_vs_z", 9.5, 5.0),
    _sim("pl19_800", "transfer_matrix", 9.8, 4.6),
    _sim("pl19_800", "psf_coupling", 7.6, 6.6),
    _sim("pl19_800", "modes_in_pupil", W_WIDE, 3.2),
    _sim("pl19_800", "output_face", 14.0, 5.0),
    # part 2: single wavelength
    _nb("pl19_800", "omp_sweep", "trainMLP", 14, 0, 7.8, 5.4, title=False),
    _nb("pl19_800", "mlp_training", "trainMLP", 5, 0, W_WIDE, 4.2),
    _nb("pl19_1200", "omp_sweep", "trainMLP", 14, 0, 7.8, 5.4, title=False),
    _nb("pl19_1200", "mlp_training", "trainMLP", 5, 0, W_WIDE, 4.2),
    _nb("pl19_800", "single_comparison", "workbench", 32, 0, 9.0, 4.6, title=False, legend="below", ncol=3),
    _nb("pl19_800", "forward_model_noise", "symbolic", 36, 0, W_HALF, 4.6),
    # part 3: dispersed lantern
    _sim("pl19_800", "pupil_petals", 5.6, 5.0),
    _nb("pl19_800", "sample_spectra", "workbench", 40, 0, W_HALF, 4.4, title=False),
    _nb("pl19_800", "core0_ramp", "multi_wavelength", 10, 0, 7.8, 5.2, title=False),
    _nb("pl19_1200", "fiber_spectra", "multi_wavelength", 8, 1, W_HALF, 4.9, title=False, legend="right", ncol=1),
    _nb("pl19_1200", "fiber_throughput", "multi_wavelength", 9, 2, W_HALF, 4.9, title=False, legend="right", ncol=1),
    # part 4: comparison
    _nb("pl19_800", "wrapped_rmse", "PL_EML", 5, 0, 10.0, 4.6, title=False, legend="right", ylabel="wrapped RMSE (nm)"),
    _nb("pl19_800", "unwrap_scatter", "PL_EML", 7, 0, 14.0, 3.9, **SCATTER),
    _nb("pl19_800", "spectral_comparison", "workbench", 57, 0, 9.0, 4.6, title=False, legend="below", ncol=3),
    *[_nb(L, "single_comparison", "workbench", 32, 0, W_HALF, 4.6, title=False, legend="below", ncol=2)
      for L in ("pl7_1200", "pl19_1200")],
    # single wavelength with a wavefront error besides the pistons (f^-2, models trained at each level)
    *[_nb(L, "wfe", "trainMLP", 23, 0, 7.6, 5.6, title=False, legend="below", ncol=2) for L in ("pl7_1200", "pl19_1200")],
    # the same comparison with the errors taken modulo lambda (third figure of the cell)
    _nb("pl19_1200", "sr_fronts", "workbench", 25, 0, 7.2, 5.0, title=False, legend="below", ncol=2),
    _nb("pl19_1200", "cubic_svd", "trainMLP", 14, 1, 6.4, 4.6, title=False),
    # per-wavelength models trained with 30 nm of wavefront error + unwrap (wfe_unwrap.ipynb)
    _nb("pl19_1200", "wfe_wrapped", "wfe_unwrap", 3, 0, 11.0, 4.4, title=False, legend="right", ylabel="wrapped RMSE (nm)"),
    _nb("pl19_1200", "wfe_unwrap_scatter", "wfe_unwrap", 4, 0, 14.0, 3.9, **SCATTER),
    _nb("pl19_1200", "wfe_err_dist", "wfe_unwrap", 6, 0, W_FULL, 4.4, title=False,
        legend=None),
    _nb("pl19_1200", "wfe_spectral_comparison", "wfe_unwrap", 5, 0, W_HALF, 4.6, title=False, legend="below", ncol=2),
    _nb("pl19_1200", "wfe_spectral_dof", "wfe_unwrap", 5, 1, W_HALF, 4.6, title=False, no_tidy=True),
    _nb("pl19_1200", "cf_convergence", "workbench", 23, 0, 11.0, 3.5, title=False, dense_y=True),
    _nb("pl19_1200", "sr_scatter", "workbench", 25, 1, 13.0, 3.1, **SCATTER),
    # the final comparisons: per-piston bars sorted by decreasing error, and mean error vs dof
    _nb("pl7_1200", "single_dof", "workbench", 32, 3, W_HALF, 4.6, title=False),
    _nb("pl19_1200", "single_dof", "workbench", 32, 3, W_HALF, 4.6, title=False,
        labels=dict(linear=(-0.015, 0.05, 'right', 'bottom'),
                                            **{'symbolic regression': (0.015, -0.06, 'left', 'top'),
                                               'dense poly, degree 2': (0.02, 0.04, 'left', 'bottom'),
                                               'dense poly, degree 3': (-0.02, 0.05, 'right', 'bottom'),
                                               'sparse poly (OMP), K=1282': (-0.02, 0.05, 'right', 'bottom'),
                                               'closed form': (0.02, 0.03, 'left', 'bottom'),
                                               'MLP': (-0.02, 0.04, 'right', 'bottom')})),
    *[_nb(L, "spectral_dof", "workbench", 57, 1, W_HALF, 4.6, title=False, no_tidy=True)
      for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "single_comparison_mod", "workbench", 32, 2, W_HALF, 4.6, title=False, legend="below", ncol=2)
      for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "spectral_comparison", "workbench", 57, 0, W_HALF, 4.6, title=False, legend="below", ncol=2)
      for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "wrapped_rmse", "PL_EML", 5, 0, 10.0, 4.6, title=False, legend="right", ylabel="wrapped RMSE (nm)") for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "unwrap_scatter", "PL_EML", 7, 0, 14.0, 3.9, **SCATTER) for L in ("pl7_1200", "pl19_1200")],
    ("ambiguity_vs_range", (("sim", "ambiguity_vs_range"), W_WIDE, 5.4, {})),
    _nb("pl19_800", "hpo_grid", "mlp_hpo", 9, 0, 9.5, 4.2),
    # appendix
    _nb("pl19_800", "psf_zoo_piston", "psf_zoo", 6, 0, 6.0, 6.5),
    _nb("pl19_800", "psf_zoo_twins", "psf_zoo", 9, 0, 7.0, 6.5),
    _nb("pl19_800", "omp_degree_mix", "workbench", 28, 0, 9.0, 5.8, title=False),
    _nb("pl19_800", "more_data", "mlp_hpo", 18, 0, 11.0, 6.0),
    _nb("pl19_800", "phasor_r2", "workbench", 41, 0, 11.0, 5.0, title=False, legend="right"),
    *[_sim(L, f, W_HALF, 4.4) for L in ("pl7_1200", "pl19_1200") for f in ("geometry", "neff_vs_z")],
    *[_sim(L, "psf_coupling", W_HALF, 5.6) for L in ("pl7_1200", "pl19_1200")],
    *[_sim(L, "output_face", 14.0, 4.4) for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "core0_ramp", "multi_wavelength", 10, 0, 6.8, 4.6, title=False) for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "sample_spectra", "workbench", 40, 0, 6.2, 2.4, title=False, ylabel="power") for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "coupling_vs_wl", "multi_wavelength", 12, 0, 6.2, 2.4, title=False, ylabel="coupling") for L in ("pl7_1200", "pl19_1200")],
])


# --------------------------------------------------------------------------- #
def code_ordinal(notebook_path, cell):
    """The pickles are named by execution order: the n-th non-empty code cell
    (1-based), as nbclient runs them. Cell index -> that n."""
    nb = json.load(open(notebook_path, encoding="utf-8"))
    n = 0
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "code" and "".join(c["source"]).strip():
            n += 1
        if i == cell:
            assert c["cell_type"] == "code", f"{notebook_path} cell {cell} is not code"
            return n
    raise IndexError(f"{notebook_path} has no cell {cell}")


def source_path(src):
    if src[0] == "sim":
        return os.path.join(PKL, src[1] + ".pkl")
    _, run, notebook, cell, k = src
    rundir = os.path.join(os.environ.get("SLIDEFIG_RUNS", os.path.join(PLREC, "runs")), run)
    nb = os.path.join(rundir, notebook + ".ipynb")
    if not os.path.exists(nb):
        return None
    hits = sorted(glob.glob(os.path.join(rundir, "figs", f"{notebook}_*_{k}.pkl")))
    n = code_ordinal(nb, cell)
    for h in hits:                       # <notebook>_x<NNN>_<k>.pkl or <notebook>_<cell id>_<k>.pkl
        tag = os.path.basename(h)[len(notebook) + 1:-len(f"_{k}.pkl")]
        if tag == f"x{n:03d}" or tag == nb_cell_id(nb, cell):
            return h
    return None


def nb_cell_id(notebook_path, cell):
    return json.load(open(notebook_path, encoding="utf-8"))["cells"][cell].get("id")


def restyle(fig, w_cm, h_cm, legend=None, ncol=None, title=True, suptitle=False, retitle=(),
            short=True, relabel=(), share=False, nbins=None, xlabel=None, ylabel=None,
            font=FONT_PT, dense_y=False, labels=None):
    """Resize to the slot and set every text to one size, then lay out again.
    Lines, markers and scatter points shrink with the figure (they were sized
    for the notebook's larger figure), within readable minima."""
    s = min(1.0, max(0.35, w_cm * CM / fig.get_size_inches()[0]))
    if dense_y:                       # more labelled ticks on the first axes only
        for ax in fig.axes[:1]:
            if ax.get_yscale() == "log":
                # two labels per decade (1 and 3): denser without the label pile-up
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 3.0), numticks=30))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(),
                                                      numticks=30))
            else:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=9, prune=None))
                ax.yaxis.set_minor_locator(AutoMinorLocator(2))
                ax.tick_params(axis="y", which="minor", length=1.6, width=0.4)
    fig.set_size_inches(w_cm * CM, h_cm * CM)
    fig.set_dpi(300)
    # the deck does not show the no-skill / chance reference: drop it everywhere
    _DROP = ("no skill", "no-skill", "noskill", "chance", "no model")
    for ax in fig.axes:
        for art in (list(getattr(ax, "lines", [])) + list(getattr(ax, "collections", []))
                    + list(getattr(ax, "patches", [])) + list(getattr(ax, "texts", []))):
            try:
                lab = (art.get_label() or "").lower()
            except Exception:
                lab = ""
            try:
                txt = (art.get_text() or "").lower()
            except Exception:
                txt = ""
            if any(k in lab or k in txt for k in _DROP):
                try:
                    art.remove()
                except Exception:
                    pass
        # bar charts: the label sits on the BarContainer, not on its rectangles
        for cont in list(getattr(ax, "containers", [])):
            if any(k in (cont.get_label() or "").lower() for k in _DROP):
                for p in list(cont):
                    try:
                        p.remove()
                    except Exception:
                        pass
                ax.containers.remove(cont)
    for ln in fig.findobj(Line2D):
        ln.set_linewidth(max(0.5, ln.get_linewidth() * s))
        ln.set_markersize(max(2.0, ln.get_markersize() * s))
        ln.set_markeredgewidth(max(0.3, ln.get_markeredgewidth() * s))
    for c in fig.findobj(Collection):
        if isinstance(c, PathCollection) and len(c.get_sizes()):
            c.set_sizes(np.maximum(1.0, np.asarray(c.get_sizes()) * s ** 2))
        lw = np.asarray(c.get_linewidths(), float)
        if lw.size:
            c.set_linewidths(np.maximum(0.2, lw * s))
    for p in fig.findobj(Patch):
        if p.get_linewidth():
            p.set_linewidth(max(0.2, p.get_linewidth() * s))
    for t in fig.findobj(Text):
        t.set_fontsize(font)
    # titles: the slide has a frame title and a caption, so the figure's suptitle
    # goes (unless suptitle=True), and the axes titles too with title=False;
    # retitle: [(regex, replacement)] applied to the axes titles that stay
    if fig._suptitle is not None and not suptitle:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        if not title:
            ax.set_title("")
        for pat, repl in retitle:
            ax.set_title(re.sub(pat, repl, ax.get_title()))
    # axis labels replaced (shorter ones for small panels)
    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        if xlabel is not None and ax.get_xlabel():
            ax.set_xlabel(xlabel)
        if ylabel is not None and ax.get_ylabel():
            ax.set_ylabel(ylabel)
    # one shared x / y label for a row of panels with the same labels
    if share and len(fig.axes) > 1:
        main = [ax for ax in fig.axes if ax.get_label() != "<colorbar>"]
        xl = {ax.get_xlabel() for ax in main if ax.get_xlabel()}
        yl = {ax.get_ylabel() for ax in main if ax.get_ylabel()}
        if len(xl) == 1:
            for ax in main:
                ax.set_xlabel("")
            fig.supxlabel(xl.pop(), fontsize=font)
        if len(yl) == 1:
            x0 = min(ax.get_position().x0 for ax in main)
            for ax in main:
                ax.set_ylabel("")
                if ax.get_position().x0 > x0 + 1e-3:
                    ax.tick_params(labelleft=False)
            fig.supylabel(yl.pop(), fontsize=font)
    if nbins:
        for ax in fig.axes:
            if ax.get_label() == "<colorbar>":
                continue
            for axis in ("x", "y"):
                if getattr(ax, f"get_{axis}scale")() == "linear" and \
                        type(getattr(ax, f"{axis}axis").get_major_locator()).__name__ in ("AutoLocator", "MaxNLocator"):
                    ax.locator_params(axis=axis, nbins=nbins)
    # legends: shortened labels; moved out of the axes ("below" / "right"), or kept
    for ax in fig.axes:
        lg = ax.get_legend()
        if lg is None:
            continue
        handles, labels = lg.legend_handles, [t.get_text() for t in lg.get_texts()]
        if short:
            labels = [re.sub(r"\s*\([^()]*dof/piston\)", "", s) for s in labels]
        for pat, repl in relabel:
            labels = [re.sub(pat, repl, s) for s in labels]
        keep = [i for i, s_ in enumerate(labels) if not any(k in s_.lower() for k in _DROP)]
        if legend is None and len(keep) == len(labels):
            for t, s_ in zip(lg.get_texts(), labels):
                t.set_text(s_)
            continue
        handles, labels = [handles[i] for i in keep], [labels[i] for i in keep]
        lg.remove()
        if legend is None:                            # entries dropped: rebuild in place
            ax.legend(handles, labels, fontsize=font)
            continue
        kw = dict(fontsize=font, frameon=False, handlelength=1.2, columnspacing=1.0, handletextpad=0.4)
        if legend == "below":
            fig.legend(handles, labels, loc="outside lower center", ncol=ncol or len(labels), **kw)
        elif legend == "right":
            fig.legend(handles, labels, loc="outside right center", ncol=ncol or 1, **kw)
    for lg in fig.findobj(Legend):
        lg.set_title(lg.get_title().get_text() or None, prop={"size": font})
    for t in fig.findobj(Text):                       # again: set_title & co. reset the size
        t.set_fontsize(font)
    try:
        fig.set_layout_engine("constrained", h_pad=0.02, w_pad=0.02, hspace=0.04, wspace=0.04)
    except RuntimeError:              # colorbars made without constrained layout: keep a compatible engine
        try:
            fig.set_layout_engine("tight", pad=0.3)
        except RuntimeError:
            pass
    return fig


def render(name, quiet=False):
    src, w, h, opts = SLOTS[name]
    path = source_path(src)
    if path is None or not os.path.exists(path):
        old = os.path.join(HERE, "figures", "nb", name + ".png")
        if FALLBACK and os.path.exists(old):           # the first campaign's PNG, fitted to the slot
            img = plt.imread(old)
            ih, iw = img.shape[:2]
            fw, fh = w, h
            if iw / ih > w / h:
                fh = w * ih / iw
            else:
                fw = h * iw / ih
            fig = plt.figure(figsize=(fw * CM, fh * CM))
            fig.add_axes([0, 0, 1, 1]).imshow(img, interpolation="antialiased")
            fig.axes[0].axis("off")
            os.makedirs(OUT, exist_ok=True)
            fig.savefig(os.path.join(OUT, name + ".pdf"), dpi=300)
            fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
            plt.close(fig)
            print(f"  ~~ {name}: no pickle, old PNG fitted to {fw:.1f} x {fh:.1f} cm")
            return True
        if not quiet:
            print(f"  -- {name}: no pickle ({src})")
        return False
    with open(path, "rb") as f:
        fig = pickle.load(f)
    if src[0] == "nb":
        canonical_colours(name, fig)      # one colour per method, whatever the draw order
    if src[0] == "sim":            # drawn at slide size by make_figures / make_ambiguity_figure: as is
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
        plt.close(fig)
        print(f"  ok {name}  (as drawn)")
        return True
    restyle(fig, w, h, **{k: v for k, v in opts.items()
                          if k not in ('no_tidy', 'labels')})
    if opts.get('labels'):
        place_labels(fig, opts['labels'])
    elif not opts.get('no_tidy'):
        tidy_labels(fig)                     # smaller, offset method labels where a plot has them
    # the saved (tight) box must fit the slot: shrink the canvas until it does
    for _ in range(6):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        bw, bh = bb.width / CM, bb.height / CM
        if bw <= w * 1.02 and bh <= h * 1.02:
            break
        cw, ch = fig.get_size_inches()
        fig.set_size_inches(cw * min(1.0, w / bw), ch * min(1.0, h / bh))
    os.makedirs(OUT, exist_ok=True)
    kw = dict(bbox_inches="tight", pad_inches=0.02)      # nothing cut; the size stays ~ the slot
    fig.savefig(os.path.join(OUT, name + ".pdf"), **kw)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=150, **kw)
    plt.close(fig)
    print(f"  ok {name}  ({w:g} x {h:g} cm)")
    return True


def lp_name(props, n, radius):
    """LP_lm name (TeX) of mode n of a guide's cross-section, read from the field itself.

    props: {"points": (M, 2) node coordinates, "modes": (N, M) mode values}; radius: the
    multimode core radius. l is the dominant azimuthal harmonic on rings inside the core,
    m - 1 the number of sign changes of that harmonic along the radius. For the guided modes
    of the lanterns' multimode end it reproduces the ideal step-index order (LP01, LP11, LP21,
    LP02, LP31, ...); near-cut-off modes, spread into the jacket, may not be readable."""
    from scipy.interpolate import LinearNDInterpolator
    f = LinearNDInterpolator(np.asarray(props["points"]), np.asarray(props["modes"])[n])
    tt = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    harm = lambda r: np.fft.rfft(np.nan_to_num(f(r * np.cos(tt), r * np.sin(tt))))   # noqa: E731
    l = int(np.argmax(sum(np.abs(harm(r)[:10]) ** 2 for r in np.linspace(0.2, 0.9, 8) * radius)))
    prof = np.array([harm(r)[l] for r in np.linspace(0.02, 1.0, 300) * radius])
    proj = np.real(prof * np.exp(-1j * np.angle(prof[np.argmax(np.abs(prof))])))
    proj = proj[np.abs(proj) > 1e-3 * np.abs(proj).max()]
    return f"LP$_{{{l}{1 + int(np.sum(np.diff(np.sign(proj)) != 0))}}}$"


def lp_names(props, modes, radius, all_modes=None):
    """{n: LP name} for the modes n in ``modes``; the two members of a degenerate pair (the
    cos / sin orientations of an LP_lm with l >= 1) get distinct names, LP_lma and LP_lmb, in
    index order. all_modes: the modes over which pairs are counted (default ``modes``), so that
    a mode keeps the same name in figures showing different subsets."""
    allm = sorted(set(all_modes if all_modes is not None else modes) | set(modes))
    base = {n: lp_name(props, n, radius) for n in allm}
    seen, out = {}, {}
    for n in allm:
        b = base[n]
        k = seen.get(b, 0); seen[b] = k + 1
        pair = sum(1 for m in allm if base[m] == b) > 1
        out[n] = b.replace("}$", ("\\mathrm{" + "ab"[k] + "}}$") if pair and k < 2 else "}$")
    return {n: out[n] for n in modes}


def export(fig, name, native=False):
    """For the figure scripts: pickle the figure (figures/pkl), then write the
    slide version -- as drawn if native (already at slide size and style),
    else redrawn to its slot."""
    os.makedirs(PKL, exist_ok=True)
    with open(os.path.join(PKL, name + ".pkl"), "wb") as f:
        pickle.dump(fig, f)
    if native:
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
        w, h = fig.get_size_inches() / CM
        print(f"  ok {name}  ({w:.1f} x {h:.1f} cm, native)")
    elif name in SLOTS:
        render(name)


if __name__ == "__main__":
    args = sys.argv[1:]
    if args == ["--list"]:
        for k, (src, w, h, o) in SLOTS.items():
            print(f"{k:32s} {w:5.1f} x {h:4.1f} cm  {src}  {source_path(src) or ''}")
        sys.exit()
    FALLBACK = "--fallback" in args             # slots without a pickle: the old PNG, fitted
    args = [a for a in args if a != "--fallback"]
    names = args or list(SLOTS)
    done = sum(render(n) for n in names)
    print(f"{done} / {len(names)} slide figures written to {OUT}")
