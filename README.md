# PLREC

Wavefront reconstruction from photonic-lantern signals: DM-mode coefficients
(here the 5 petal pistons of the ANDES modal basis) estimated from the power
in the 19 output cores of a 19-port lantern, at one wavelength or across a
spectrum. The training data are simulated with
[cbeam](https://github.com/jw-lin/cbeam).

| | |
|---|---|
| `ml_reconstruct/` | datasets, JAX MLP, sparse (OMP) and dense polynomial fits, Zernike features, multi-wavelength phase unwrap, fit-or-load model cache |
| `ml_reconstruct/simulate.py` | dataset generation with cbeam's `batch_pipeline` / `multi_wvl` examples |
| `trainMLP.ipynb` | one wavelength (800 nm), pistons within +-lambda/2: MLP vs. linear, k-NN, polynomial (dense / OMP), Zernike features, optional symbolic regression |
| `trainMLP_multiwavelength.ipynb` | +-5 um pistons from the 41-wavelength (700-900 nm) spectral response: single models on the whole spectrum |
| `PL19_EML.ipynb` | the same data: one model per wavelength on the wrapped piston, fused by a multi-wavelength unwrap |
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

Datasets and fitted models are written to `data/ml/` (not tracked). The
notebooks simulate a dataset only if its file is missing:
`trainMLP.ipynb` makes `single_800nm.npz` (10000 samples, about 40 minutes),
`trainMLP_multiwavelength.ipynb` makes `spectral_41wl.npz` (50000 samples x
41 wavelengths, several hours), which `PL19_EML.ipynb` reuses. Models are
cached by name in `data/ml/cache_*/`: delete those folders after
regenerating a dataset.

Moved out of cbeam's examples (branch `ml-reconstruction`, commit c2ca638):
cbeam keeps only the propagation.
