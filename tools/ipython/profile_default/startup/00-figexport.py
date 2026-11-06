"""Figure export for the slides -- active only with PLREC_FIG_EXPORT=<dir>.

run_lantern.sh and leonardo/extras.slurm execute the notebooks with
IPYTHONDIR=PLREC/tools/ipython, so every kernel runs this file at start. Each
matplotlib figure a notebook draws is pickled when it is closed (after the
inline backend has displayed it) as

    <dir>/<PLREC_FIG_PREFIX>_<cell id>_<k>.pkl     (k: 0, 1, ... within the cell)

Presentation/Slides/slide_figs.py then redraws the chosen ones at the size of
their slot on the slide, with one font size for all. The notebook's own output
is not changed.
"""
import os


def _figexport():
    out = os.environ.get("PLREC_FIG_EXPORT")
    if not out:
        return
    import pickle
    import weakref
    from IPython import get_ipython
    from matplotlib._pylab_helpers import Gcf

    ip = get_ipython()
    if ip is None:
        return
    os.makedirs(out, exist_ok=True)
    prefix = os.environ.get("PLREC_FIG_PREFIX", "nb")
    state = {"cell": "x000", "n": 0, "k": 0}
    seen = weakref.WeakSet()

    def pre_run_cell(info):
        state["n"] += 1
        state["k"] = 0
        state["cell"] = getattr(info, "cell_id", None) or f"x{state['n']:03d}"

    def dump(fig):
        if fig in seen:
            return
        seen.add(fig)
        name = f"{prefix}_{state['cell']}_{state['k']}"
        state["k"] += 1
        try:
            data = pickle.dumps(fig)
        except Exception as e:                      # e.g. a lambda tick formatter
            try:
                import cloudpickle
                data = cloudpickle.dumps(fig)
            except Exception:
                print(f"[figexport] {name}: not picklable ({e})")
                return
        with open(os.path.join(out, name + ".pkl"), "wb") as f:
            f.write(data)

    destroy, destroy_all = Gcf.destroy, Gcf.destroy_all

    def _destroy(num):
        m = num if hasattr(num, "canvas") else Gcf.figs.get(num)
        if m is not None:
            dump(m.canvas.figure)
        destroy(num)

    def _destroy_all():
        for m in list(Gcf.figs.values()):
            dump(m.canvas.figure)
        destroy_all()

    Gcf.destroy, Gcf.destroy_all = _destroy, _destroy_all
    ip.events.register("pre_run_cell", pre_run_cell)


_figexport()
del _figexport
