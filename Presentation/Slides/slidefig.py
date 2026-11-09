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
from matplotlib.collections import Collection, PathCollection   # noqa: E402
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
    # part 4: comparison
    _nb("pl19_800", "wrapped_rmse", "PL_EML", 5, 0, 10.0, 4.6, title=False, legend="right", ylabel="wrapped RMSE (nm)"),
    _nb("pl19_800", "unwrap_scatter", "PL_EML", 7, 0, 14.0, 3.9, **SCATTER),
    _nb("pl19_800", "spectral_comparison", "workbench", 57, 0, 9.0, 4.6, title=False, legend="below", ncol=3),
    *[_nb(L, "single_comparison", "workbench", 32, 0, W_HALF, 4.6, title=False, legend="below", ncol=2)
      for L in ("pl7_1200", "pl19_1200")],
    # single wavelength with a wavefront error besides the pistons (f^-2, models trained at each level)
    *[_nb(L, "wfe", "trainMLP", 23, 0, 7.6, 5.6, title=False, legend="below", ncol=2) for L in ("pl7_1200", "pl19_1200")],
    # the same comparison with the errors taken modulo lambda (third figure of the cell)
    _nb("pl19_1200", "sr_fronts", "workbench", 25, 0, W_WIDE, 5.0, title=False, legend="below", ncol=2),
    # the final comparisons: per-piston bars sorted by decreasing error, and mean error vs dof
    *[_nb(L, "single_dof", "workbench", 32, 3, W_HALF, 4.6, title=False) for L in ("pl7_1200", "pl19_1200")],
    *[_nb(L, "spectral_dof", "workbench", 57, 1, W_HALF, 4.6, title=False) for L in ("pl7_1200", "pl19_1200")],
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
            short=True, relabel=(), share=False, nbins=None, xlabel=None, ylabel=None, font=FONT_PT):
    """Resize to the slot and set every text to one size, then lay out again.
    Lines, markers and scatter points shrink with the figure (they were sized
    for the notebook's larger figure), within readable minima."""
    s = min(1.0, max(0.35, w_cm * CM / fig.get_size_inches()[0]))
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
    if src[0] == "sim":            # drawn at slide size by make_figures / make_ambiguity_figure: as is
        os.makedirs(OUT, exist_ok=True)
        fig.savefig(os.path.join(OUT, name + ".pdf"))
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=150)
        plt.close(fig)
        print(f"  ok {name}  (as drawn)")
        return True
    restyle(fig, w, h, **opts)
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
