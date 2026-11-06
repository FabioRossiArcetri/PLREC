# Runbook: simulating a lantern and running the notebooks

One `cbeam` checkout and one `PLREC` checkout serve every lantern design. A
design is a named **preset** in cbeam's `examples/batch_pipeline/constants.py`
(`LANTERNS`), selected with the environment variable `CBEAM_LANTERN`:

| preset | ports | design / band | notes |
|---|---|---|---|
| `pl19_800` (default) | 19 | 800 nm / 700-900 nm | the reference design; caches tagged `19port_XXXX` |
| `pl7_1200` | 7 | 1200 nm / 1100-1300 nm | Ge-doped cores, F-doped capillary, multimode NA 0.170, `rclad` 6.4 um |
| `pl19_1200` | 19 | 1200 nm / 1100-1300 nm | same fibres as `pl7_1200`, `rclad` 10 um (V = 8.90 as the reference) |

All three keep the PSF matched to the multimode end the same way (fill =
0.30 x V_ref / V; exactly 0.30 for `pl19_800`). The per-preset mode
bookkeeping tables are in the same file; `pl19_1200` infers its bookkeeping
per wavelength while characterizing, as its weakest modes merge near 1300 nm.

PLREC's per-preset settings (piston range, polynomial budgets, data folder)
are in `ml_reconstruct/presets.py`. Data: `data/ml/` for `pl19_800` (its
original location), `data/ml/<preset>/` for the others. Executed notebooks:
`runs/<preset>/`.

## On sauron

The compute copies are `/raid1/frossi/dev/work/{cbeam,PLREC}`: plain copies
of the Windows working trees (the git checkouts in `/raid1/frossi/dev/` are
left alone), refreshed with `sync_to_sauron.sh` from the Windows side.

```bash
source /opt/miniconda3/etc/profile.d/conda.sh; conda activate base14
export PYTHONPATH=/raid1/frossi/dev/work/cbeam/src     # base14's installed cbeam is another checkout
export CBEAM_LANTERN=pl19_1200
```

Every heavy step keeps to **half of sauron's memory**: it first measures the
peak memory of one job, then runs as many in parallel as fit in
`MEM_BUDGET_GB` (default: half the RAM) with a 30% margin. Long jobs are
detached: `setsid nohup bash <script> > <log> 2>&1 < /dev/null &`.

### 1. Characterize the 41 wavelengths

```bash
cd /raid1/frossi/dev/work/cbeam && bash run_characterize.sh     # logs/<preset>/
```

Caches: `examples/data/` (tags from `batch_pipeline.characterization_tag`),
with the bookkeeping decisions of each wavelength in `examples/data/bookkeeping/`.

### 2. Generate the datasets

```bash
cd /raid1/frossi/dev/work/PLREC && bash run_gen.sh              # logs/<preset>/
```

Single-wavelength dataset (10000 samples) plus 410 spectral pieces (41
wavelengths x 10 chunks of 5000), then assembled into
`data/ml/<preset>/spectral_41wl.npz`. Each piece runs single-threaded
(`CBEAM_FFT_WORKERS=1`: scipy's FFT ignores `OMP_NUM_THREADS`).

### 3. Run the notebooks

```bash
cd /raid1/frossi/dev/work/PLREC && mkdir -p runs/$CBEAM_LANTERN
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir runs/$CBEAM_LANTERN \
  trainMLP.ipynb trainMLP_multiwavelength.ipynb PL_EML.ipynb multi_wavelength.ipynb workbench.ipynb
```

(cbeam's `examples/` notebooks run the same way into `examples/runs/<preset>/`.)

## Results of the first campaign (2026-11-06, sauron)

With the earlier piston ranges: +-300 nm (800 nm) / +-500 nm (1200 nm) at one
wavelength, +-5 um dispersed. The presets now use +-lambda/2 and +-8 lambda for
every lantern (`ml_reconstruct/presets.py`); the Leonardo rerun produces the
results for those.

| RMSE, validation set | pl19_800 | pl7_1200 | pl19_1200 |
|---|---|---|---|
| **one wavelength**: range | +-300 nm (0.375 lambda) | +-500 nm (0.42 lambda) | +-500 nm (0.42 lambda) |
| no skill | 173 nm | 288 nm | 288 nm |
| linear | 103.1 nm | 261.9 nm | 213.3 nm |
| dense quadratic | 56.5 nm | 243.2 nm | 117.1 nm |
| OMP, cubic (K = 1282 / 119 / 1282) | 8.4 nm | 228.5 nm | 31.9 nm |
| MLP (3 x 128) | 9.0 nm | 226.4 nm | 25.9 nm |
| **41 wavelengths**, pistons +-5 um: no skill | 2893 nm | 2893 nm | 2893 nm |
| full-spectrum MLP (4 x 256, 10000 epochs) | 10.25 nm | 278.0 nm | **9.48 nm** |
| per-wavelength OMP + unwrap | 1.64 nm | 1665 nm (53% wrong fringes) | 30.0 nm (0.08% wrong fringes) |
| flat-wavefront coupling over the band | 34-53% | 41-57% | 36-50% |
| signal collisions at +-0.42 lambda | 4.3% | 37% | 7.6% |

Why the 7-port does so much worse: an N-port lantern gives N - 1 independent
normalised signals per wavelength, and a smooth map of the 5 pistons is in
general one-to-one over a large range only into >= 2 x 5 + 1 = 11 dimensions;
18 (19 ports) qualify, 6 (7 ports) do not, so the signals fold and different
piston sets coincide (collision test: for each of 3000 random configurations,
the one with the closest signals has pistons > lambda/4 rms away;
`Presentation/Slides/diag_collisions.py`). pl19_1200 -- same fibres and
wavelength as pl7_1200, 19 ports -- confirms it: 7.6% vs 37% collisions, and
9.5 vs 278 nm with the full-spectrum MLP. The pl7_1200 redesign (rclad 6.2 ->
6.4 um) changed little (221.8 -> 226.4 nm single, 257.8 -> 278.0 nm spectral).
pl19_1200's per-wavelength unwrap is limited by ~1 in 1000 wrong-fringe
outliers (piston 2 without them: 0.65 nm).

## Not ported (pl19_800 only)

`derivation`, `symbolic`, `psf_zoo`, `piston2`, `mlp_hpo` (the former
workbench Part C) and the `scripts/` they use are specific to the reference
lantern.
