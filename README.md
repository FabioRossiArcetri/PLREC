# PLREC

Wavefront reconstruction from photonic-lantern signals: DM-mode coefficients
(here the 5 petal pistons of the ANDES modal basis) estimated from the power
in the output cores of a lantern, at one wavelength or across a spectrum. The
training data are simulated with [cbeam](https://github.com/jw-lin/cbeam).

**Lanterns.** The pipeline notebooks run for any lantern preset of cbeam's
`batch_pipeline.LANTERNS`, chosen with `CBEAM_LANTERN`: `pl19_800` (the
reference 19-port lantern at 800 nm, default), `pl7_1200` (7 ports at
1200 nm) and `pl19_1200` (19 ports at 1200 nm, same fibres as `pl7_1200`).
Per-preset settings are in `ml_reconstruct/presets.py`, data in `data/ml/`
(`pl19_800`) or `data/ml/<preset>/`, executed notebooks in `runs/<preset>/`.
`RUNBOOK.md` has the commands (characterization, datasets, notebooks, within
half of sauron's memory) and the results so far.

| | |
|---|---|
| `ml_reconstruct/` | datasets, JAX MLP, sparse (OMP) and dense polynomial fits, Zernike features, multi-wavelength phase unwrap, fit-or-load model cache |
| `ml_reconstruct/simulate.py` | dataset generation with cbeam's `batch_pipeline` / `multi_wvl` examples |
| `trainMLP.ipynb` | one wavelength, pistons within +-lambda/2: MLP vs. linear, k-NN, polynomial (OMP), Zernike diagnostic, optional symbolic regression |
| `trainMLP_multiwavelength.ipynb` | +-8 lambda pistons from the 41-wavelength spectral response: single models on the whole spectrum |
| `PL_EML.ipynb` | the same data: one model per wavelength on the wrapped piston, fused by a multi-wavelength unwrap |
| `workbench.ipynb` | the three notebooks above combined, as the working notebook |
| `multi_wavelength.ipynb` | the lantern's spectral response to a +-8 lambda ramp of one petal piston, through the same pipeline as the spectral dataset (from cbeam's examples, with `multi_wavelength_vis.py`; its results are cached in `<data>/mw_ramp_41.npz`) |
| `gen_datasets.py`, `run_gen.sh` | both datasets of a lantern, chunked and in parallel within a memory budget |
| `scripts/mlp_hpo.py` | small MLP hyperparameter search (size vs. accuracy) at one wavelength and at 20 / 10 / 5 nm wavelength steps, reported in `mlp_hpo.ipynb` (pl19_800 only, formerly workbench Part C); `scripts/run_mlp_hpo.sh` runs it on one GPU |
| `mlp_hpo.ipynb` | the MLP hyperparameter search results (pl19_800 only) |
| `piston2.ipynb` | why piston 2 is the easiest piston at one wavelength: the pupil's twin symmetry and the reference petal, not the lantern's clocking; its simulations and trainings are `scripts/piston2_twins.py` / `scripts/piston2_reref.py` (run by `scripts/run_piston2.sh`) |
| `psf_zoo.ipynb` | a zoo of PSFs at the lantern input (flat, single petal pistons, twin pairs, low-order aberrations, SCAO-like residuals) with the lantern's outputs; `scripts/psf_zoo.py` simulates those |
| `symbolic.ipynb` | analytical and symbolic reconstructors at one wavelength: the exact closed-form forward model (619 constants) and its Gauss-Newton inversion, and explicit expressions from symbolic regression (PySR, Operon); `scripts/symbolic_sr.py` (run by `scripts/run_symbolic_sr.sh` / `run_symbolic_sr_range.sh`) and `scripts/symbolic_residuals.py` produce its inputs |
| `derivation.ipynb` | the single-wavelength closed form derived step by step with sympy (pupil -> FFT -> modal projection -> propagation -> detection -> normalisation), and its constants computed from first principles with cbeam (helpers in `ml_reconstruct/derivation.py`) |
| `tests/` | `pytest tests` |

## Requirements

numpy, scipy, matplotlib, **jax + optax**, **scikit-learn**; the optional
symbolic-regression cell of `trainMLP.ipynb` needs **PySR** (and Julia).

Generating the datasets also needs a cbeam checkout with the
multi-wavelength examples: `ml_reconstruct.simulate` imports `batch_pipeline`
and `multi_wvl` from its `examples/` folder and loads the lantern
characterizations cached in `examples/data/` there (characterizing the
lantern if they are missing: tens of minutes per wavelength). Set
`CBEAM_EXAMPLES` to that folder; the default is `../cbeam/examples`, next to
this repository.

## Data

Datasets and fitted models are written to the lantern's data folder (not
tracked): `data/ml/` for `pl19_800`, `data/ml/<preset>/` otherwise. The
notebooks simulate a dataset only if its file is missing: `trainMLP.ipynb`
makes the single-wavelength one (10000 samples, 15-40 minutes),
`trainMLP_multiwavelength.ipynb` the spectral one (50000 samples x 41
wavelengths, many hours in one process), which `PL_EML.ipynb` reuses;
`gen_datasets.py` / `run_gen.sh` make both in parallel. Models are cached by
name in `<data>/cache_*/`: delete those folders after regenerating a
dataset.

Moved out of cbeam's examples (branch `ml-reconstruction`, commit c2ca638):
cbeam keeps only the propagation.
