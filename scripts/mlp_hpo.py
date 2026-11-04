"""Small MLP hyperparameter search: accuracy vs. size, MLP only.

Two approaches, in these scenarios (inputs -> 5 petal pistons):

    800nm    one wavelength, +-300 nm (single_800nm.npz): inside the capture range
    1wl-800  one wavelength of the spectral dataset (800 nm), +-5 um: far outside it
    20nm     the spectral dataset at 20 nm steps (11 wavelengths), +-5 um
    10nm     10 nm steps (21 wavelengths)
    5nm      5 nm steps (all 41 wavelengths)

* full spectrum: one MLP on the channel powers of all the scenario's
  wavelengths at once, regressing the pistons directly;
* per wavelength + unwrap (spectral scenarios): one small MLP per wavelength
  on its 19 channel powers, regressing the phasor (sin, cos)(2 pi p / lambda)
  of each piston -- the piston wrapped at that wavelength -- and the
  multi-wavelength unwrap of ml_reconstruct.unwrap fusing the wavelengths of
  the scenario. The 20 and 10 nm wavelengths are subsets of the 41, so one set
  of 41 per-wavelength models serves all three scenarios.

Stages:

    grid    full spectrum: depth {2, 3, 4} x width {32, 64, 128, 256}, Adam with a
            cosine-decaying learning rate from 3e-4, in each scenario but
            1wl-800; plus reference runs with the notebooks' recipe (constant
            3e-4: 3 x 128 / 500 epochs at 800 nm, 4 x 256 / 10000 epochs at
            5 nm) and one 4 x 256 run in 1wl-800.
    perwl   per wavelength: PERWL_ARCHS at each of the 41 wavelengths.
    seeds   full spectrum: the best and the "balanced" configuration of each
            grid scenario (the fewest parameters within BALANCE_TOL of the
            best), retrained with two more seeds, to measure the training
            variance.
    unwrap  per wavelength + unwrap: every PERWL_ARCHS x spectral scenario,
            on the held-out and the validation rows.
    data800 more data at 800 nm: the best 800nm configuration (4 x 256) on the
            first DATA800_SIZES rows of the 50000-sample dataset
            (single_800nm_50k.npz, made by data/gen/gen_single_50k.py), with
            DATA800_SEEDS seeds each -- a learning curve, and the ensemble of
            the seeds at each size. All sizes share the held-out and the
            validation rows of that dataset.

Protocol: the last ES_SHARE of the training split is held out to pick the best
epoch and the configurations; the validation split of the dataset is touched
only to report the final RMSE (it plays the role of a test set, and is the
split the notebooks score on). Each run appends one JSON line to
data/ml/mlp_hpo/<stage>_w<worker>.jsonl and saves its weights and
predictions to data/ml/mlp_hpo/models/<run_id>.npz; finished runs are
skipped, so the script can be restarted.

    python scripts/mlp_hpo.py {grid,perwl,seeds,unwrap,data800,list} [--worker K --workers N]

(scripts/run_mlp_hpo.sh runs the stages with several workers on one GPU.)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, ROOT)
import ml_reconstruct as ml  # noqa: E402

OUT = os.path.join(ROOT, "data", "ml", "mlp_hpo")
DATASETS = {"single": os.path.join(ROOT, "data", "ml", "single_800nm.npz"),
            "spectral": os.path.join(ROOT, "data", "ml", "spectral_41wl.npz"),
            "single50k": os.path.join(ROOT, "data", "ml", "single_800nm_50k.npz")}
SPAN_NM = 5000.0                 # piston range of the spectral dataset
ES_SHARE = 0.1                   # share of the training split held out for model selection
BALANCE_TOL = 1.10               # "balanced": fewest parameters within 10% of the best RMSE
DEPTHS, WIDTHS = (2, 3, 4), (32, 64, 128, 256)
RECIPES = {"cos3e-4": dict(learning_rate=3e-4, lr_schedule="cosine"),
           "const3e-4": dict(learning_rate=3e-4, lr_schedule="constant")}
SCENARIOS = {            # dataset, wavelength indices into the 41 (None: single dataset), epochs
    "800nm":   dict(dataset="single", wl=None, epochs=2000),
    "1wl-800": dict(dataset="spectral", wl=[20], epochs=5000),
    "20nm":    dict(dataset="spectral", wl=list(range(0, 41, 4)), epochs=5000),
    "10nm":    dict(dataset="spectral", wl=list(range(0, 41, 2)), epochs=5000),
    "5nm":     dict(dataset="spectral", wl=list(range(41)), epochs=5000),
    "800nm-50k": dict(dataset="single50k", wl=None, epochs=2000),
}
GRID_SCENARIOS = ("800nm", "20nm", "10nm", "5nm")
UNWRAP_SCENARIOS = ("20nm", "10nm", "5nm")
SEEDS = (1, 2)
PERWL_ARCHS = ((2, 32), (2, 64), (3, 64), (3, 128))     # (depth, width)
PERWL_EPOCHS = 2000
N_WL = 41
DATA800_SIZES = (7200, 18000, 36000)   # training rows; 7200 = the 10000-sample dataset's
DATA800_SEEDS = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- #
# run definitions
# --------------------------------------------------------------------------- #
def run(scenario, depth, width, recipe="cos3e-4", epochs=None, seed=0, role="grid"):
    """A full-spectrum run."""
    return dict(kind="full", scenario=scenario, depth=depth, width=width, recipe=recipe,
                epochs=epochs or SCENARIOS[scenario]["epochs"], seed=seed, role=role)


def perwl_run(w, depth, width):
    """A per-wavelength run, at wavelength index w."""
    return dict(kind="perwl", scenario=f"wl{w:02d}", wl_index=w, depth=depth, width=width,
                recipe="cos3e-4", epochs=PERWL_EPOCHS, seed=0, role="perwl")


def data800_run(n_fit, seed):
    """A run on the first n_fit training rows of the 50000-sample 800 nm dataset."""
    return dict(run(f"800nm-50k", 4, 256, seed=seed, role="data800"), n_fit=n_fit)


def run_id(r):
    n = f"-n{r['n_fit']}" if r.get("n_fit") else ""
    return f"{r['scenario']}-d{r['depth']}w{r['width']}-{r['recipe']}-e{r['epochs']}{n}-s{r['seed']}"


def layers_of(r):
    if r["kind"] == "perwl":
        return [19] + [r["width"]] * r["depth"] + [10]
    wl = SCENARIOS[r["scenario"]]["wl"]
    return [19 * (1 if wl is None else len(wl))] + [r["width"]] * r["depth"] + [5]


def n_macs(layers):
    """Multiply-adds of one forward pass (the dense layers; ~1 per weight)."""
    return int(sum(a * b for a, b in zip(layers[:-1], layers[1:])))


def grid_runs():
    runs = [run(s, d, w) for s in GRID_SCENARIOS for d in DEPTHS for w in WIDTHS]
    runs += [run("800nm", 3, 128, "const3e-4", 500, role="reference"),
             run("5nm", 4, 256, "const3e-4", 10000, role="reference"),
             run("1wl-800", 4, 256, role="single-wl")]
    return runs


def perwl_runs():
    return [perwl_run(w, d, wd) for d, wd in PERWL_ARCHS for w in range(N_WL)]


def data800_runs():
    return [data800_run(n, s) for n in DATA800_SIZES for s in DATA800_SEEDS]


def load_results(stage=None):
    rows = []
    for path in sorted(glob.glob(os.path.join(OUT, f"{stage or '*'}_w*.jsonl"))):
        with open(path) as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def pick(rows):
    """(best, balanced) grid rows of one scenario, by the held-out (es) RMSE."""
    rows = [r for r in rows if r["role"] == "grid"]
    best = min(rows, key=lambda r: r["es_rmse_mean"])
    ok = [r for r in rows if r["es_rmse_mean"] <= BALANCE_TOL * best["es_rmse_mean"]]
    balanced = min(ok, key=lambda r: (r["n_params"], r["es_rmse_mean"]))
    return best, balanced


def seed_runs():
    grid = load_results("grid")
    runs = []
    for s in GRID_SCENARIOS:
        rows = [r for r in grid if r["scenario"] == s]
        if not rows:
            continue
        for tag, r in zip(("best", "balanced"), pick(rows)):
            for seed in SEEDS:
                cand = run(s, r["depth"], r["width"], r["recipe"], r["epochs"], seed, role=f"seed-{tag}")
                if run_id(cand) not in {run_id(x) for x in runs}:
                    runs.append(cand)
    return runs


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
_DS = {}


def dataset(name):
    if name not in _DS:
        _DS[name] = ml.LanternDataset.load(DATASETS[name])
    return _DS[name]


def split(ds):
    """(signals, pistons) of the fit, held-out (es) and validation rows."""
    S_tr, C_tr = ds.train
    S_va, C_va = ds.val
    n_fit = int(round((1 - ES_SHARE) * len(S_tr)))
    return (S_tr[:n_fit], C_tr[:n_fit]), (S_tr[n_fit:], C_tr[n_fit:]), (S_va, C_va)


def data(r):
    """Standardised inputs and the targets of a run: pistons standardised
    (full spectrum) or phasors at the run's wavelength (per wavelength)."""
    if r["kind"] == "perwl":
        ds, w = dataset("spectral"), r["wl_index"]
        feat = lambda S: S[:, w]
    else:
        sc = SCENARIOS[r["scenario"]]
        ds = dataset(sc["dataset"])
        feat = (lambda S: S) if sc["wl"] is None else (lambda S: S[:, sc["wl"]].reshape(len(S), -1))
    (S_fit, C_fit), (S_es, C_es), (S_va, C_va) = split(ds)
    if r.get("n_fit"):                  # a learning-curve run: the first n_fit fit rows only
        S_fit, C_fit = S_fit[:r["n_fit"]], C_fit[:r["n_fit"]]
    xs = ml.Standardizer().fit(feat(S_fit))
    X = [xs.transform(feat(S)) for S in (S_fit, S_es, S_va)]
    if r["kind"] == "perwl":
        T = ds.wavelengths_nm[r["wl_index"]]
        to_y, from_y = (lambda C: ml.phasor_targets(C, T)), (lambda Y: Y)
    else:
        ys = ml.Standardizer().fit(C_fit)
        to_y, from_y = ys.transform, ys.inverse
    return dict(X_fit=X[0], X_es=X[1], X_va=X[2], Y_fit=to_y(C_fit), Y_es=to_y(C_es),
                C_es=C_es, C_va=C_va, from_y=from_y)


# --------------------------------------------------------------------------- #
# training stages
# --------------------------------------------------------------------------- #
def execute(r, stage, worker):
    d = data(r)
    layers = layers_of(r)
    params, hist = ml.train_mlp(layers, d["X_fit"], d["Y_fit"], d["X_es"], d["Y_es"],
                                epochs=r["epochs"], init_seed=r["seed"], train_seed=100 + r["seed"],
                                log_every=0, **RECIPES[r["recipe"]])
    out_es = d["from_y"](np.asarray(ml.mlp_forward(params, d["X_es"])))
    out_va = d["from_y"](np.asarray(ml.mlp_forward(params, d["X_va"])))
    row = dict(r, run_id=run_id(r), layers=layers, n_params=ml.n_params(layers), n_macs=n_macs(layers),
               best_epoch=hist["best_epoch"], train_seconds=round(hist["seconds"], 1))
    if r["kind"] == "perwl":            # wrapped-piston RMSE at this wavelength
        T = dataset("spectral").wavelengths_nm[r["wl_index"]]
        es, va = (np.sqrt(np.mean(ml.wrap(ml.phasor_decode(out, T)[0] - ml.wrap(C, T), T) ** 2, axis=0))
                  for out, C in ((out_es, d["C_es"]), (out_va, d["C_va"])))
        row["wavelength_nm"] = float(T)
    else:
        es, va = ml.rmse_nm(out_es, d["C_es"]), ml.rmse_nm(out_va, d["C_va"])
    row.update(es_rmse=es.round(3).tolist(), es_rmse_mean=round(float(es.mean()), 3),
               val_rmse=va.round(3).tolist(), val_rmse_mean=round(float(va.mean()), 3))
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    ml.save_mlp(os.path.join(OUT, "models", run_id(r) + ".npz"), params, layers,
                train=np.asarray(hist["train"], np.float32), val=np.asarray(hist["val"], np.float32),
                best_epoch=hist["best_epoch"], pred_es=out_es.astype(np.float32),
                pred_va=out_va.astype(np.float32))
    with open(os.path.join(OUT, f"{stage}_w{worker}.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[w{worker}] {row['run_id']:40s} {row['n_params']:7d} params  es {row['es_rmse_mean']:8.2f}  "
          f"val {row['val_rmse_mean']:8.2f} nm  best epoch {row['best_epoch']:5d}  "
          f"{row['train_seconds']:6.0f} s", flush=True)


def cost(r):
    """Rough relative training cost, to spread the runs evenly over the workers."""
    n_wl = 1 if r["kind"] == "perwl" else len(SCENARIOS[r["scenario"]]["wl"] or [0])
    return r["epochs"] * (n_wl * 50 + n_macs(layers_of(r)) / 1e3) * (r.get("n_fit") or 1)


# --------------------------------------------------------------------------- #
# unwrap stage
# --------------------------------------------------------------------------- #
def unwrap_rows():
    """Per wavelength + unwrap: every PERWL_ARCHS x UNWRAP_SCENARIOS
    combination whose per-wavelength runs have all finished."""
    perwl = {r["run_id"]: r for r in load_results("perwl")}
    ds = dataset("spectral")
    _, (_, C_es), (_, C_va) = split(ds)
    done = {r["run_id"] for r in load_results("unwrap")}
    for depth, width in PERWL_ARCHS:
        for scen in UNWRAP_SCENARIOS:
            rid = f"{scen}-unwrap-d{depth}w{width}-cos3e-4-e{PERWL_EPOCHS}-s0"
            wl = SCENARIOS[scen]["wl"]
            members = [run_id(perwl_run(w, depth, width)) for w in wl]
            if rid in done or not all(m in perwl for m in members):
                continue
            t0 = time.time()
            periods = ds.wavelengths_nm[wl]
            sig = np.array([perwl[m]["es_rmse"] for m in members])       # (n_wl, M) wrapped RMSE
            score = {}
            for split_name, C in (("es", C_es), ("va", C_va)):
                outs = [ml.load_mlp(os.path.join(OUT, "models", m + ".npz"))[2][f"pred_{split_name}"]
                        for m in members]
                dec = [ml.phasor_decode(o, T) for o, T in zip(outs, periods)]
                fit = dict(ghat_va=np.stack([g for g, _ in dec], axis=2),
                           conf_va=np.stack([c for _, c in dec], axis=2), periods=periods, sig_tr=sig)
                P_hat, _ = ml.unwrap_all(fit, SPAN_NM)
                score[split_name] = (ml.rmse_nm(P_hat, C),
                                     float(np.mean(np.abs(P_hat - C) > periods.min() / 2)))
                if split_name == "va":
                    np.save(os.path.join(OUT, "models", rid + "-pred_va.npy"), P_hat.astype(np.float32))
            layers = layers_of(perwl_run(0, depth, width))
            yield dict(kind="perwl+unwrap", scenario=scen, depth=depth, width=width, recipe="cos3e-4",
                       epochs=PERWL_EPOCHS, seed=0, role="unwrap", run_id=rid, layers=layers,
                       n_models=len(wl), n_params=len(wl) * ml.n_params(layers),
                       n_macs=len(wl) * n_macs(layers),
                       train_seconds=round(sum(perwl[m]["train_seconds"] for m in members), 1),
                       es_rmse=score["es"][0].round(3).tolist(),
                       es_rmse_mean=round(float(score["es"][0].mean()), 3),
                       val_rmse=score["va"][0].round(3).tolist(),
                       val_rmse_mean=round(float(score["va"][0].mean()), 3),
                       es_wrong_fringe=score["es"][1], val_wrong_fringe=score["va"][1],
                       unwrap_seconds=round(time.time() - t0, 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", choices=["grid", "perwl", "seeds", "unwrap", "data800", "list"])
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.stage == "list":
        for r in grid_runs() + perwl_runs():
            print(run_id(r), layers_of(r), ml.n_params(layers_of(r)))
        return
    if a.stage == "unwrap":
        for row in unwrap_rows():
            with open(os.path.join(OUT, f"unwrap_w{a.worker}.jsonl"), "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"{row['run_id']:40s} {row['n_params']:8d} params  es {row['es_rmse_mean']:8.2f}  "
                  f"val {row['val_rmse_mean']:8.2f} nm  wrong fringe {100 * row['val_wrong_fringe']:.2f}%",
                  flush=True)
        return
    runs = {"grid": grid_runs, "perwl": perwl_runs, "seeds": seed_runs, "data800": data800_runs}[a.stage]()
    runs = sorted(runs, key=cost, reverse=True)
    done = {r["run_id"] for r in load_results()}
    mine = [r for i, r in enumerate(runs) if i % a.workers == a.worker and run_id(r) not in done]
    print(f"[w{a.worker}] {a.stage}: {len(mine)} runs to do", flush=True)
    for r in mine:
        execute(r, a.stage, a.worker)


if __name__ == "__main__":
    main()
