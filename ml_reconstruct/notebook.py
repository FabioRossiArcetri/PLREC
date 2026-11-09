"""Helpers for the notebooks (workbench.ipynb): scoring, comparison plots and
fit-or-load wrappers around the models of this package.

Not imported by ``ml_reconstruct`` itself, so the core package does not need
matplotlib: ``from ml_reconstruct import notebook as nb``.
"""
from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from .data import LanternDataset, rmse_nm
from .mlp import load_mlp, save_mlp, train_mlp
from .sparse import omp_fit, omp_predict
from .unwrap import phasor_targets
from .presets import settings


# --------------------------------------------------------------------------- #
# datasets
# --------------------------------------------------------------------------- #
def load_or_simulate(path, maker, **kwargs):
    """The LanternDataset saved at ``path``. If it is missing, it is made by
    ``ml_reconstruct.simulate.<maker>(**kwargs)`` and saved there first; cbeam
    is imported only then."""
    if not os.path.exists(path):
        from . import simulate
        ds = getattr(simulate, maker)(**kwargs)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ds.save(path)
    return LanternDataset.load(path)


def square_pixel_aspect(extent, shape):
    """imshow ``aspect`` that makes the pixels of an image of ``shape`` (rows,
    cols) square when it is drawn with ``extent`` [x0, x1, y0, y1] in data
    units: the pixel width over the pixel height."""
    x0, x1, y0, y1 = extent
    return abs((x1 - x0) / shape[1]) / abs((y1 - y0) / shape[0])


def plot_single_overview(S, C, core_xy):
    """Single-wavelength dataset: piston histograms, the spread of each core's
    signal, and the mean and relative variation of the signals per core."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4))
    ax[0].hist(C, bins=40, histtype="step", label=[f"piston {k}" for k in range(C.shape[1])])
    ax[0].set_xlabel("piston (nm)"); ax[0].set_ylabel("samples"); ax[0].legend(fontsize=7)
    ax[0].set_title("piston coefficients")
    ax[1].boxplot(S, positions=np.arange(S.shape[1]), showfliers=False)
    ax[1].set_xlabel("core"); ax[1].set_ylabel("signal"); ax[1].set_title("core signals (whiskers: 1.5 IQR)")
    plt.tight_layout(); plt.show()
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    hexmap(ax[0], core_xy, S.mean(0), "mean signal")
    hexmap(ax[1], core_xy, S.std(0) / S.mean(0), "std / mean", cmap="magma")
    plt.tight_layout(); plt.show()


def plot_signal_piston_correlation(S, C):
    """Correlation of each core signal with each piston p (the odd part of the
    response) and with p^2 (the even part)."""
    n_c, M = S.shape[1], C.shape[1]
    corr = np.corrcoef(np.c_[S, C, C ** 2].T)
    fig, ax = plt.subplots(1, 2, figsize=(16, 3.4))
    for a, c, t in [(ax[0], corr[:n_c, n_c:n_c + M], "corr(signal, p)"),
                    (ax[1], corr[:n_c, n_c + M:], "corr(signal, p^2)")]:
        im = a.imshow(c.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        a.set_xticks(range(n_c)); a.set_yticks(range(M), [f"piston {k}" for k in range(M)])
        a.set_xlabel("core"); a.set_title(t); plt.colorbar(im, ax=a, fraction=0.02)
    plt.tight_layout(); plt.show()


def plot_total_power(lam, S, ramp_file=None):
    """Total channel power vs. wavelength of a spectral dataset S (n, n_wl,
    n_cores): mean and +-1 std over the samples, and the flat-wavefront curve
    of multi_wavelength.ipynb's ramp results if ``ramp_file`` exists.
    Returns the mean total per wavelength."""
    tot = S.sum(axis=2)
    m, s = tot.mean(0), tot.std(0)
    plt.figure(figsize=(8, 4))
    plt.fill_between(lam, m - s, m + s, alpha=0.25, label="+-1 std, random pistons")
    plt.plot(lam, m, "o-", ms=3, label="mean, random pistons")
    if ramp_file and os.path.exists(ramp_file):
        with np.load(ramp_file) as f:
            ramp, ramp_wl = f["power_spectra"], f["wavelengths"]
        plt.plot(ramp_wl, ramp[:, ramp.shape[1] // 2].sum(axis=1), "k--", label="flat wavefront")
    plt.xlabel("wavelength (nm)"); plt.ylabel("fraction of incident power")
    plt.title("total channel power"); plt.legend(fontsize=8); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    return m


def plot_sample_spectra(lam, S, C, i=0):
    """The channel spectra of sample i of a spectral dataset."""
    n_cores = S.shape[2]
    plt.figure(figsize=(8, 4))
    for c in range(n_cores):
        plt.plot(lam, S[i, :, c], lw=1, color=plt.cm.turbo(c / (n_cores - 1)))
    plt.xlabel("wavelength (nm)"); plt.ylabel("fraction of incident power"); plt.grid(alpha=0.3)
    plt.title(f"sample {i}: the {n_cores} channel spectra (pistons {np.round(C[i]).astype(int)} nm)",
              fontsize=9)
    plt.tight_layout(); plt.show()


def plot_spectral_maps(lam, S):
    """Mean channel power and its relative variation per (wavelength, core)."""
    ext = [lam[0] - (lam[1] - lam[0]) / 2, lam[-1] + (lam[1] - lam[0]) / 2, -0.5, S.shape[2] - 0.5]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for a, v, t in [(ax[0], S.mean(0).T, "mean channel power"), (ax[1], (S.std(0) / S.mean(0)).T, "std / mean")]:
        im = a.imshow(v, aspect=square_pixel_aspect(ext, v.shape), origin="lower", extent=ext)
        a.set_xlabel("wavelength (nm)"); a.set_ylabel("core"); a.set_title(t); plt.colorbar(im, ax=a)
    plt.tight_layout(); plt.show()


def phasor_r2(X3, C, lam):
    """(n_wl, M) R^2 of a linear fit of each piston's phasor (sin, cos)(2 pi p /
    lambda) on the channel powers X3 (n, n_wl, n_cores) at that wavelength,
    averaged over sin and cos."""
    M = C.shape[1]
    R2 = np.zeros((len(lam), M))
    for w in range(len(lam)):
        A = np.c_[X3[:, w], np.ones(len(X3))]
        T = phasor_targets(C, lam[w])
        resid = T - A @ np.linalg.lstsq(A, T, rcond=None)[0]
        r2 = 1 - resid.var(0) / T.var(0)
        R2[w] = 0.5 * (r2[:M] + r2[M:])
    return R2


def plot_per_piston(x, Y, xlabel, ylabel, title, logy=False, ref=None, ref_label=None):
    """One line per piston (columns of Y) against x, with an optional dashed
    reference curve."""
    plt.figure(figsize=(9, 4))
    for k in range(Y.shape[1]):
        plt.plot(x, Y[:, k], "o-", ms=3, label=f"piston {k}")
    if ref is not None:
        plt.plot(x, ref, "k--", lw=1, label=ref_label)
    if logy:
        plt.yscale("log")
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# scoring and comparison
# --------------------------------------------------------------------------- #
def score(results, name, C_pred, C_true, ys=None, note="", period=None, wrapped=None):
    """Store and print the per-piston RMSE [nm] of C_pred against C_true.

    ys : the target Standardizer when C_pred is in standardised units (it is
        inverted first); None when C_pred is already in nm.
    period, wrapped : at one wavelength, the wavelength [nm] and a dict that
        receives the RMSE of the errors taken modulo it (rmse_nm(..., period)),
        also printed.
    """
    if ys is not None:
        C_pred = ys.inverse(C_pred)
    r = results[name] = rmse_nm(C_pred, C_true)
    print(f"{name:22s} mean {r.mean():8.2f} nm   per piston {np.round(r, 2)}{note}")
    if period is not None:
        rw = rmse_nm(C_pred, C_true, period)
        if wrapped is not None:
            wrapped[name] = rw
        print(f"{'  modulo lambda':22s} mean {rw.mean():8.2f} nm   per piston {np.round(rw, 2)}")
    return r


def n_dof(model, outputs_per_piston=1):
    """Fitted degrees of freedom **per piston**: ``outputs_per_piston`` (2 for
    phasor targets, PhasorCodec) times the degrees of freedom per output."""
    return outputs_per_piston * _n_dof(model)


def _n_dof(model):
    """Fitted degrees of freedom **per output**: every model here is fitted
    independently for each piston, so this counts one piston's parameters and
    never their sum over the five pistons -- the non-zero coefficients and the
    intercept of a linear or OMP fit, an MLP's parameters divided by its number
    of outputs (the hidden layers are shared by all pistons), and for k-NN the
    stored training values of one piston."""
    from sklearn.linear_model import RidgeCV
    from sklearn.neighbors import KNeighborsRegressor
    if isinstance(model, KNeighborsRegressor):
        # one piston keeps the training inputs plus its own target
        return int(model.n_samples_fit_ * (model.n_features_in_ + 1))
    if isinstance(model, dict):                                   # MLP parameters
        total = int(sum(np.size(a) for v in model.values() for a in v))
        biases = model.get("biases")
        n_out = int(np.size(biases[-1])) if biases else 1
        return total / max(n_out, 1)
    if isinstance(model, tuple) and len(model) == 3:              # (poly, scaler, ridge or OMP fits)
        return _n_dof(model[2])
    if isinstance(model, RidgeCV):
        return int(np.shape(model.coef_)[-1] + 1)                 # one row of coef_ is one piston
    if isinstance(model, tuple) and len(model) == 2:              # one (omp, sel) fit: already per piston
        return int(np.count_nonzero(model[0].coef_)) + 1
    if isinstance(model, list):                                   # one model per piston
        return float(np.mean([_n_dof(m) for m in model]))
    if hasattr(model, "coef_"):                                   # e.g. ElasticNetCV (multi-output)
        c = np.asarray(model.coef_)
        rows = c.reshape(-1, c.shape[-1]) if c.ndim > 1 else c.reshape(1, -1)
        return float(np.mean([np.count_nonzero(r) for r in rows]) + 1)
    raise TypeError(f"no degree-of-freedom count for {type(model).__name__}")


def show_results(results, title, ref=None, logy=True, sort=False, dof=None, exclude=(),
                 ylabel="validation RMSE (nm)"):
    """Table and grouped bar chart of a results dict {name: per-piston RMSE};
    ref: the name whose mean is drawn as a dashed line; sort: order the models
    by decreasing mean RMSE; dof: {name: degrees of freedom PER PISTON},
    shown in the table and the legend, and against the mean RMSE in a second plot;
    exclude: models kept in the table but left out of the plots."""
    if sort:
        results = dict(sorted(results.items(), key=lambda kv: -kv[1].mean()))
    dof = dof or {}
    lab = {m: f"{m} ({dof[m]:,.0f} dof/piston)" if m in dof else m for m in results}
    for m, r in results.items():
        print(f"  {m:22s} {r.mean():8.2f} nm   per piston {np.round(r, 2)}"
              + (f"   {dof[m]:>9,.0f} dof/piston" if m in dof else ""))
    names = [m for m in results if m not in exclude]
    n_modes = len(next(iter(results.values())))
    x, w = np.arange(n_modes), 0.8 / len(names)
    plt.figure(figsize=(12, 4.8))
    for i, m in enumerate(names):
        plt.bar(x + (i - len(names) / 2 + 0.5) * w, results[m], w, label=lab[m])
    if ref is not None and ref not in exclude:
        plt.axhline(results[ref].mean(), color="k", ls="--", lw=1)
    if logy:
        plt.yscale("log")
    plt.xticks(x, [f"piston {k}" for k in x]); plt.ylabel(ylabel)
    plt.title(title); plt.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plt.grid(True, axis="y", ls=":", alpha=0.5); plt.tight_layout(); plt.show()
    if dof:
        plt.figure(figsize=(7, 4.5))
        for i, m in enumerate(names):
            if m in dof:
                plt.plot(dof[m], results[m].mean(), "o", ms=8, color=plt.cm.tab10(i % 10))
                plt.annotate(m + (" (stored values)" if m == "kNN" else ""), (dof[m], results[m].mean()),
                             fontsize=8, xytext=(5, 3), textcoords="offset points")
        plt.xscale("log")
        if logy:
            plt.yscale("log")
        plt.xlabel("degrees of freedom per piston"); plt.ylabel("mean " + ylabel)
        plt.title("accuracy vs. degrees of freedom per piston"); plt.grid(alpha=0.3, which="both")
        plt.tight_layout(); plt.show()


def hexmap(ax, core_xy, values, title, **scatter_kw):
    """One value per core, drawn at the core positions on the unit disk."""
    xy = core_xy / np.linalg.norm(core_xy, axis=1).max()
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=350, **scatter_kw)
    for i, (x, y) in enumerate(xy):
        ax.text(x, y, str(i), ha="center", va="center", fontsize=7, color="w")
    ax.set_aspect("equal"); ax.set_title(title); plt.colorbar(sc, ax=ax)


# --------------------------------------------------------------------------- #
# MLP
# --------------------------------------------------------------------------- #
def fit_or_load_mlp(path, layers, Xtr, Ytr, Xva, Yva, epochs, lr, retrain=False, **train_kw):
    """Train an MLP (keeping the epoch with the lowest validation loss) and save
    it to ``path``, or load it from there. Returns ``(params, history)``."""
    if os.path.exists(path) and not retrain:
        params, _, extra = load_mlp(path)
        hist = {"train": extra["train"], "val": extra["val"], "best_epoch": int(extra["best_epoch"])}
    else:
        params, hist = train_mlp(layers, Xtr, Ytr, Xva, Yva, epochs=epochs, learning_rate=lr, **train_kw)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_mlp(path, params, layers, train=hist["train"], val=hist["val"],
                 best_epoch=hist["best_epoch"])
    print(f"MLP {layers}: best epoch {hist['best_epoch']} of {len(hist['val'])}")
    return params, hist


def plot_mlp(hist, C_true, C_pred, k, span, s=4):
    """Loss curves (dotted: best epoch), and estimate vs truth for piston k."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist["train"], label="train"); ax[0].plot(hist["val"], label="validation")
    ax[0].axvline(hist["best_epoch"], color="k", ls=":", lw=1)
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("Huber loss"); ax[0].legend()
    ax[1].scatter(C_true[:, k], C_pred[:, k], s=s, alpha=0.4)
    ax[1].plot([-span, span], [-span, span], "r--")
    ax[1].set_xlabel("true piston (nm)"); ax[1].set_ylabel("MLP estimate (nm)"); ax[1].set_title(f"piston {k}")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# MLP hyperparameter search (scripts/mlp_hpo.py)
# --------------------------------------------------------------------------- #
def load_hpo_runs(directory):
    """Every finished run of scripts/mlp_hpo.py in ``directory``, one row each
    (a pandas DataFrame), with a readable ``config`` label."""
    import glob
    import json
    import pandas as pd
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    if len(df):
        df["config"] = [f"{d}x{w} {rc} {e}ep" + (" per-wl" if k != "full" else "")
                        for d, w, rc, e, k in zip(df.depth, df.width, df.recipe, df.epochs, df.kind)]
    return df


def hpo_predictions(row, directory):
    """Saved validation estimates [nm] of one run (full spectrum) or one
    per-wavelength + unwrap combination."""
    if row["kind"] == "perwl+unwrap":
        return np.load(os.path.join(directory, "models", row["run_id"] + "-pred_va.npy"))
    return load_mlp(os.path.join(directory, "models", row["run_id"] + ".npz"))[2]["pred_va"]


def hpo_seed_summary(df, C_val, directory, gross_nm=350.0):
    """Per scenario: the picked full-spectrum configurations (best / balanced,
    from the seed stage) over their seeds, the reference runs, and the
    per-wavelength + unwrap architecture with the lowest held-out RMSE.

    C_val : {scenario: (n_val, M) true validation pistons [nm]}, to count the
        gross errors (|error| > gross_nm) from the saved predictions.
    """
    import pandas as pd
    groups = []
    picks = df[df.role.str.startswith("seed-")]
    for (scen, role), g in picks.groupby(["scenario", "role"]):
        cfg = g.config.iloc[0]
        groups.append((scen, role[5:], pd.concat([df[(df.scenario == scen) & (df.config == cfg)
                                                     & (df.seed == 0) & (df.role == "grid")], g])))
    for _, r in df[df.role.isin(["reference", "single-wl"])].iterrows():
        groups.append((r.scenario, r.role, df.loc[[r.name]]))
    for scen, g in df[df.role == "unwrap"].groupby("scenario"):
        groups.append((scen, "per-wl + unwrap", g.loc[[g.es_rmse_mean.idxmin()]]))
    rows = []
    for scen, pick, runs in groups:
        v = np.stack(runs.val_rmse.to_numpy())
        err = [np.abs(hpo_predictions(r, directory) - C_val[scen]) for _, r in runs.iterrows()]
        rows.append(dict(scenario=scen, pick=pick, config=runs.config.iloc[0],
                         n_params=int(runs.n_params.iloc[0]), seeds=len(runs),
                         val_rmse=v.mean(1).mean(), seed_min=v.mean(1).min(), seed_max=v.mean(1).max(),
                         per_piston=np.round(v.mean(0), 1), train_s=runs.train_seconds.mean(),
                         **{"gross_%": 100 * np.mean([np.mean(e > gross_nm) for e in err])}))
    return pd.DataFrame(rows)


def plot_hpo_grid(df, scenarios, titles=None):
    """Validation RMSE vs. number of parameters, one panel per scenario: the
    full-spectrum grid runs (one line per depth, labelled by width), the
    reference run (star), the picked configurations (circled), and the
    per-wavelength + unwrap architectures (black diamonds; parameters summed
    over the wavelengths)."""
    fig, axes = plt.subplots(1, len(scenarios), figsize=(4.2 * len(scenarios), 3.8), squeeze=False)
    for ax, scen, title in zip(axes[0], scenarios, titles or scenarios):
        g = df[(df.scenario == scen) & (df.role == "grid")].sort_values("n_params")
        for depth, gd in g.groupby("depth"):
            ax.plot(gd.n_params, gd.val_rmse_mean, "o-", ms=4, label=f"depth {depth}")
            for _, r in gd.iterrows():
                ax.annotate(str(r.width), (r.n_params, r.val_rmse_mean), fontsize=6,
                            xytext=(2, 2), textcoords="offset points")
        ref = df[(df.scenario == scen) & (df.role == "reference")]
        ax.plot(ref.n_params, ref.val_rmse_mean, "k*", ms=11, label="notebook recipe")
        for role, c in [("seed-best", "r"), ("seed-balanced", "g")]:
            p = g[g.config.isin(df[(df.scenario == scen) & (df.role == role)].config.unique())]
            ax.plot(p.n_params, p.val_rmse_mean, "o", mfc="none", mec=c, ms=12, mew=1.5, label=role[5:])
        u = df[(df.scenario == scen) & (df.role == "unwrap")].sort_values("n_params")
        ax.plot(u.n_params, u.val_rmse_mean, "kD--", ms=5, lw=1, label="per-wl + unwrap")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("parameters"); ax.set_title(title, fontsize=10)
    axes[0, 0].set_ylabel("validation RMSE (nm), seed 0")
    axes[0, -1].legend(fontsize=7)
    plt.tight_layout(); plt.show()


def hpo_unwrap_table(df):
    """The per-wavelength + unwrap results, one row per (scenario, architecture)."""
    cols = ["scenario", "config", "n_models", "n_params", "es_rmse_mean", "val_rmse_mean",
            "val_wrong_fringe", "val_rmse"]
    u = df[df.role == "unwrap"] if len(df) else df
    return u.reindex(columns=cols).sort_values(["scenario", "n_params"]).reset_index(drop=True)


def plot_perwl_wrapped(df):
    """Wrapped-piston validation RMSE (mean over the pistons) of every
    per-wavelength MLP, one line per architecture, with the chance level."""
    g = df[df.role == "perwl"]
    if not len(g):
        print("no per-wavelength runs yet"); return
    plt.figure(figsize=(9, 4))
    for (d, w), ga in g.groupby(["depth", "width"]):
        ga = ga.sort_values("wavelength_nm")
        plt.plot(ga.wavelength_nm, ga.val_rmse_mean, "o-", ms=3,
                 label=f"{d}x{w} ({int(ga.n_params.iloc[0])} params)")
    lam = np.sort(g.wavelength_nm.unique())
    plt.plot(lam, lam / np.sqrt(12), "k--", lw=1, label="chance")
    plt.yscale("log"); plt.xlabel("wavelength (nm)"); plt.ylabel("wrapped-piston RMSE (nm)")
    plt.title("per-wavelength MLPs"); plt.legend(fontsize=8); plt.grid(alpha=0.3, which="both")
    plt.tight_layout(); plt.show()


def data800_table(df, C_val, directory):
    """The more-data stage at 800 nm: per training size, the single-model
    validation RMSE over the seeds and that of their ensemble (the mean of the
    seeds' estimates)."""
    import pandas as pd
    rows = []
    for n, g in df[df.role == "data800"].groupby("n_fit"):
        single = np.array([r.val_rmse_mean for _, r in g.iterrows()])
        ens = np.mean([hpo_predictions(r, directory) for _, r in g.iterrows()], axis=0)
        e = rmse_nm(ens, C_val)
        rows.append(dict(n_train=int(n), seeds=len(g), single=single.mean(), single_min=single.min(),
                         single_max=single.max(), ensemble=e.mean(), ensemble_per_piston=np.round(e, 2)))
    return pd.DataFrame(rows)


def error_stats(pred, true, gross_nm=50.0):
    """RMSE and the quantiles of |error| [nm], pooled over the pistons: the
    errors are heavy-tailed (a few configurations are badly wrong), so the RMSE
    alone is dominated by rare outliers and noisy on small validation sets."""
    e = np.abs(np.asarray(pred) - true)
    return dict(rmse=rmse_nm(pred, true).mean(), median=np.median(e), p90=np.percentile(e, 90),
                p99=np.percentile(e, 99), **{f"> {gross_nm:g} nm %": 100 * np.mean(e > gross_nm)},
                piston_2_rmse=rmse_nm(pred, true)[2])


def rescore_mlp(run_ids, directory, ds, rows, n_fit):
    """Mean estimate [nm] of saved MLPs (one, or an ensemble) on rows of
    dataset ``ds``, with input / target scalers fitted on its first n_fit rows
    (what scripts/mlp_hpo.py trained them with)."""
    from .data import Standardizer
    from .mlp import mlp_forward_np
    xs, ys = Standardizer().fit(ds.signals[:n_fit]), Standardizer().fit(ds.coeff[:n_fit])
    return np.mean([ys.inverse(mlp_forward_np(load_mlp(os.path.join(directory, "models", r + ".npz"))[0],
                                              xs.transform(ds.signals[rows]))) for r in run_ids], axis=0)


def plot_data800(table, ref=None):
    """Learning curve at 800 nm: single models (mean, seed range) and the
    ensemble of the seeds vs. training size; ref: (n_train, rmse, label) points."""
    plt.figure(figsize=(7, 4))
    t = table.sort_values("n_train")
    plt.errorbar(t.n_train, t.single, yerr=[t.single - t.single_min, t.single_max - t.single],
                 fmt="o-", capsize=3, label="single 4x256 MLP (seed range)")
    plt.plot(t.n_train, t.ensemble, "s-", label=f"ensemble of {int(t.seeds.max())} seeds")
    for n, r, lab in ref or []:
        plt.plot(n, r, "k*", ms=10)
        plt.annotate(lab, (n, r), fontsize=7, xytext=(4, 4), textcoords="offset points")
    plt.xscale("log"); plt.yscale("log"); plt.grid(alpha=0.3, which="both")
    plt.xlabel("training samples"); plt.ylabel("validation RMSE (nm)")
    plt.title(f"800 nm, +-{settings('pl19_800')['single_span_nm']:g} nm: more data vs. bigger models"); plt.legend(fontsize=8)
    plt.tight_layout(); plt.show()


def hpo_pivot(df, value, scenarios):
    """Grid runs as a (depth, width) x scenario table of ``value``."""
    grid = df[df.role == "grid"]
    return (grid.pivot_table(index=["depth", "width"], columns="scenario", values=value)
            .reindex(columns=[s for s in scenarios if s in set(grid.scenario)]))


def plot_hpo_summary(summary, labels):
    """Validation RMSE of the picked configurations per scenario (bars: mean
    over seeds, error bars: the seed range; labels: thousands of parameters).
    labels: {scenario: axis label}, in display order."""
    fig, ax = plt.subplots(figsize=(12, 4.6))
    picks, w = ["best", "balanced", "reference", "single-wl", "per-wl + unwrap"], 0.16
    for i, pk in enumerate(picks):
        sub = summary[summary.pick == pk].set_index("scenario")
        scen = [s for s in labels if s in sub.index]
        if not scen:
            continue
        sub, xs = sub.loc[scen], [list(labels).index(s) + (i - 2) * w for s in scen]
        ax.bar(xs, sub.val_rmse, w, label=pk, capsize=3,
               yerr=[sub.val_rmse - sub.seed_min, sub.seed_max - sub.val_rmse])
        for x, (_, r) in zip(xs, sub.iterrows()):
            ax.text(x, r.val_rmse * 1.15, f"{r.n_params / 1e3:.0f}k", ha="center", fontsize=7)
    ax.set_xticks(range(len(labels)), list(labels.values()), fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("validation RMSE (nm)")
    ax.set_title("MLP only: accuracy by wavelength sampling (labels: parameters; bars: seed range)")
    ax.legend(fontsize=8); ax.grid(True, axis="y", ls=":", alpha=0.5); plt.tight_layout(); plt.show()


def plot_hpo_estimates(rows, runs, C_val, labels, directory):
    """Estimate vs. truth, all pistons pooled, of the seed-0 run of each row
    of a hpo_seed_summary table."""
    if not len(rows):
        print("nothing to show yet"); return
    fig, ax = plt.subplots(1, len(rows), figsize=(3.3 * len(rows), 3.3), squeeze=False)
    for a, (_, r) in zip(ax[0], rows.iterrows()):
        run = runs[(runs.scenario == r.scenario) & (runs.config == r.config) & (runs.seed == 0)].iloc[0]
        pred, true = hpo_predictions(run, directory), C_val[r.scenario]
        span = np.abs(true).max()
        a.scatter(true.ravel(), pred.ravel(), s=1, alpha=0.2)
        a.plot([-span, span], [-span, span], "r--", lw=1)
        a.set_title(f"{labels[r.scenario]}, {r.pick}\n{r.config}: {rmse_nm(pred, true).mean():.1f} nm",
                    fontsize=8)
        a.set_xlabel("true piston (nm)")
    ax[0, 0].set_ylabel("MLP estimate (nm)")
    plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# polynomial models
# --------------------------------------------------------------------------- #
def omp_fits(cache, prefix, Z_tr, Y_tr, K, screen):
    """One OMP fit (at most K terms) per target column, fit-or-load as
    ``<prefix>-K<K>``."""
    return cache(f"{prefix}-K{K}", lambda: [omp_fit(Z_tr, Y_tr[:, k], K, screen)
                                            for k in range(Y_tr.shape[1])])


def omp_predict_all(Z, fits):
    """(n, n_targets) predictions of a list of OMP fits."""
    return np.stack([omp_predict(Z, f) for f in fits], axis=1)


def kept_by_degree(fits, degree):
    """(n_targets, max degree) number of kept monomials of each degree, given
    each monomial's ``degree`` (PolynomialFeatures.powers_.sum(axis=1))."""
    degs = np.arange(1, degree.max() + 1)
    kept = [sel[np.flatnonzero(omp.coef_)] for omp, sel in fits]
    return np.array([[np.sum(degree[g] == d) for d in degs] for g in kept])


def omp_sweep(cache, prefix, Z_tr, Y_tr, Z_va, C_va, ys, Ks, screen, period=None):
    """{K: (mean kept terms, mean validation RMSE [nm], the same modulo
    ``period`` or nan)} of the OMP fits at each term budget K."""
    from .sparse import n_terms
    sweep = {}
    for K in Ks:
        fits = omp_fits(cache, prefix, Z_tr, Y_tr, K, screen)
        C_hat = ys.inverse(omp_predict_all(Z_va, fits))
        sweep[K] = (np.mean([n_terms(f) for f in fits]), rmse_nm(C_hat, C_va).mean(),
                    rmse_nm(C_hat, C_va, period).mean() if period is not None else np.nan)
        print(f"K={K:5d}: {sweep[K][0]:7.1f} terms/piston   mean RMSE {sweep[K][1]:5.1f} nm"
              + (f"   modulo lambda {sweep[K][2]:5.1f} nm" if period is not None else ""))
    return sweep


def plot_omp_sweep(sweep, title):
    """Validation RMSE against the number of kept terms of an omp_sweep (and,
    when the sweep has it, the RMSE of the errors modulo the wavelength)."""
    v = np.array(list(sweep.values()))
    plt.figure(figsize=(6, 4))
    if v.shape[1] > 2 and np.isfinite(v[:, 2]).all():      # the figure of merit when it exists
        plt.semilogy(v[:, 0], v[:, 2], "ks-")
        plt.ylabel("mean validation RMSE modulo lambda (nm)")
    else:
        plt.plot(v[:, 0], v[:, 1], "ks-")
        plt.ylabel("mean validation RMSE (nm)")
    plt.xlabel("terms per fit")
    plt.title(title); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def plot_degree_mix(Ks, mix, title="degree mix of the kept terms",
                    xlabel="K (terms per piston)"):
    """Stacked share of the kept terms per degree (mix: (n_degrees, len(Ks)))."""
    plt.figure(figsize=(6, 4))
    plt.stackplot(Ks, *mix, labels=[f"degree {d + 1}" for d in range(len(mix))], alpha=0.85)
    plt.xlabel(xlabel); plt.ylabel("share of kept terms")
    plt.title(title); plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.show()


def omp_variable_importance(fits, powers):
    """(n_targets, n_variables): the total |coefficient| of the kept monomials
    each input variable appears in (powers: PolynomialFeatures.powers_)."""
    contains = (powers > 0).astype(float)
    mass = np.zeros((len(fits), powers.shape[1]))
    for k, (omp, sel) in enumerate(fits):
        nz = np.flatnonzero(omp.coef_)
        mass[k] = contains[sel[nz]].T @ np.abs(omp.coef_[nz])
    return mass


def plot_stacked_importance(mass, names, title, top=20):
    """Bar chart of the ``top`` variables by total importance, stacked per target."""
    order = np.argsort(-mass.sum(axis=0))[:top]
    plt.figure(figsize=(10, 5))
    bottom = np.zeros(len(order))
    for k in range(len(mass)):
        plt.bar([names[i] for i in order], mass[k, order], bottom=bottom, label=f"piston {k}")
        bottom += mass[k, order]
    plt.xticks(rotation=60); plt.ylabel("total |coefficient|"); plt.title(title)
    plt.legend(fontsize=8); plt.tight_layout(); plt.show()


# --------------------------------------------------------------------------- #
# per-wavelength models and unwrap
# --------------------------------------------------------------------------- #
def print_wrapped_table(fit, lam):
    """Wrapped-piston validation RMSE per wavelength of a fit_wrapped_models result."""
    M = fit["sig_va"].shape[1]
    print("wrapped-piston RMSE per wavelength [nm]  (chance ~ period / sqrt(12))")
    print("   lambda | +-T/2 | " + " ".join(f"   p{k}" for k in range(M)))
    for i, w in enumerate(fit["wl_use"]):
        print(f"  {lam[w]:7.1f} | {fit['periods'][i] / 2:5.0f} | " +
              " ".join(f"{fit['sig_va'][i, k]:5.1f}" for k in range(M)))
    print("     mean |       | " + " ".join(f"{v:5.1f}" for v in fit["sig_va"].mean(0)))


def print_unwrap_table(P_hat, C_true, fit):
    """Per piston: RMSE, bias, share of wrong fringes (error > half the shortest
    period), and the RMSE of the best single wavelength without unwrapping."""
    rmse = rmse_nm(P_hat, C_true)
    wrong = np.mean(np.abs(P_hat - C_true) > fit["periods"].min() / 2, axis=0)
    no_unwrap = rmse_nm(fit["ghat_va"][:, :, fit["sig_va"].mean(axis=1).argmin()], C_true)
    print(" piston | RMSE (nm) | bias (nm) | wrong fringe | best single wl, no unwrap")
    for k in range(len(rmse)):
        print(f"   {k}    | {rmse[k]:9.2f} | {np.mean(P_hat[:, k] - C_true[:, k]):+9.2f} | "
              f"{100 * wrong[k]:10.2f}% | {no_unwrap[k]:9.0f}")
    print(f"  mean  | {rmse.mean():9.2f} |           | {100 * wrong.mean():10.2f}% |")


def plot_unwrap(P_hat, C_true, Q, span):
    """Unwrapped estimate vs. truth per piston (colour: agreement q), and the
    histogram of the residuals."""
    M = C_true.shape[1]
    rmse = rmse_nm(P_hat, C_true)
    fig, ax = plt.subplots(1, M, figsize=(3.1 * M, 3.2), sharex=True, sharey=True)
    for k in range(M):
        sc = ax[k].scatter(C_true[:, k], P_hat[:, k], c=Q[:, k], s=3, alpha=0.4, vmin=0, vmax=1)
        ax[k].plot([-span, span], [-span, span], "r--", lw=1)
        ax[k].set_title(f"piston {k} ({rmse[k]:.1f} nm)"); ax[k].set_xlabel("true (nm)")
    ax[0].set_ylabel("unwrapped estimate (nm)")
    fig.colorbar(sc, ax=ax, label="agreement q", shrink=0.8); plt.show()
    err = (P_hat - C_true).ravel()
    plt.figure(figsize=(8, 3.4))
    plt.hist(np.clip(err, -20, 20), bins=161)
    plt.xlabel("estimate - truth (nm, clipped to +-20)")
    plt.title(f"residuals: mean |err| {np.mean(np.abs(err)):.2f} nm")
    plt.grid(alpha=0.3); plt.tight_layout(); plt.show()


def spectral_features(S, wl_idx, proj=None):
    """(n, n_wl, n_cores) signals -> (n, len(wl_idx) * d) flat features at the
    wavelength indices wl_idx: the raw core signals (proj None) or their
    projection S @ proj.T (e.g. a Zernike modal projector)."""
    S = S[:, wl_idx]
    if proj is not None:
        S = S @ proj.T
    return S.reshape(len(S), -1)


def rescore_mlp_on(run_ids, directory, ds_ref, n_fit, S_new):
    """Mean estimate [nm] of saved MLPs (one, or an ensemble) on new signals
    S_new, with the scalers they were trained with (first n_fit rows of ds_ref)."""
    from .data import Standardizer
    from .mlp import mlp_forward_np
    xs, ys = Standardizer().fit(ds_ref.signals[:n_fit]), Standardizer().fit(ds_ref.coeff[:n_fit])
    return np.mean([ys.inverse(mlp_forward_np(load_mlp(os.path.join(directory, "models", r + ".npz"))[0],
                                              xs.transform(S_new))) for r in run_ids], axis=0)
