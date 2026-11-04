"""Generate LanternDatasets by propagating DM-aberrated fields through the lantern.

Two measurement models:

* single wavelength (``simulate_core_images``): the lantern output imaged on
  a camera -- the output intensity is sampled on a regular grid, the core
  centres are calibrated on the mean image, and each core's signal is the
  mean of a small pixel window around its centre. The input field is
  normalised to unit coupled power, so the signals are the relative
  distribution of the light over the cores.
* spectral (``simulate_spectra``): the power in each output channel at every
  wavelength (MultiWavelengthPropagationPipeline), as the fraction of the
  incident power -- i.e. including the input coupling efficiency.

Columns follow the lantern's port order (``p["output_positions"]``), which is
also the order of ChainPropagator.to_channel_basis().

The propagation uses ``batch_pipeline`` and ``multi_wvl`` from a cbeam
checkout's ``examples/`` folder: ``CBEAM_EXAMPLES`` (default
``../cbeam/examples`` next to this repository). The lantern
characterizations are loaded from, and cached in, its ``data/`` folder.
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np
from scipy.interpolate import LinearNDInterpolator

CBEAM_EXAMPLES = os.path.abspath(os.environ.get(
    "CBEAM_EXAMPLES",
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, "cbeam", "examples")))
if CBEAM_EXAMPLES not in sys.path:     # appended: modules of this repository take precedence
    sys.path.append(CBEAM_EXAMPLES)

import batch_pipeline as bp  # noqa: E402

from .data import LanternDataset


def output_core_xy(p) -> np.ndarray:
    """(n_cores, 2) physical core positions [um] at the lantern output, in
    port order: the z=0 layout scaled by the taper."""
    return np.asarray(p["output_positions"], float) * p["taper_factor"]


def simulate_core_images(p, ifunc, coeff, n_train, *, prop12=None, window=5,
                         grid_resolution=bp.CALIB_GRID_RESOLUTION,
                         prop_chunk_size=None, centers_by_port=None) -> LanternDataset:
    """Single-wavelength dataset: ``coeff`` (n, n_modes) [nm] -> per-core camera
    signals (n, n_cores) in port order.

    prop12 : a characterized ChainPropagator for ``p`` (built, or loaded from
        the cache, if None).
    window : side of the pixel window averaged around each core centre, on a
        grid_resolution x grid_resolution grid over the output face.
    prop_chunk_size : propagate this many fields at a time (None: all at once).
        On the numpy backend the ODE solver keeps every step of the batch, so
        large batches need chunks to bound the memory.
    centers_by_port : (n_cores, 2) core centres [um] to place the pixel windows
        at, e.g. another dataset's (``dataset_centers``), so that the signals are
        measured exactly as there; None calibrates them on this batch's mean image.
    """
    if prop12 is None:
        with contextlib.chdir(CBEAM_EXAMPLES):     # its data/ holds the caches
            prop12 = bp.build_and_characterize_lantern(p, reuse_cache=True)
    pipeline = bp.BatchPropagationPipeline(prop12, p, ifunc)

    u0 = pipeline.generate_batch_modal_coefficients(coeff, use_gpu=False)
    uf, _, _ = pipeline.propagate_batch(u0, chunk_size=prop_chunk_size)
    intensity = np.abs(pipeline.reconstruct_batch_output_fields(uf)) ** 2   # (n, n_mesh)
    mesh = pipeline.wvg_props_output["mesh"]

    port_xy = output_core_xy(p)
    if centers_by_port is None:
        # core centres on the mean image, matched to the ports
        centers, gx, gy, dx, dy = bp.calibrate_subpixel_centers(
            intensity.mean(axis=0), mesh, grid_resolution=grid_resolution)
        port_of = bp.map_evaluated_to_ideal_geometry(centers, port_xy)   # detected k -> port
        centers_by_port = np.asarray(centers)[np.argsort(port_of)]
    else:                                   # the same grid calibrate_subpixel_centers builds
        centers_by_port = np.asarray(centers_by_port, float)
        gx = np.linspace(mesh.points[:, 0].min(), mesh.points[:, 0].max(), grid_resolution)
        gy = np.linspace(mesh.points[:, 1].min(), mesh.points[:, 1].max(), grid_resolution)
        dx, dy = gx[1] - gx[0], gy[1] - gy[0]

    # pixel windows around the centres; only those grid pixels are evaluated
    half = np.arange(window) - window // 2
    cols = np.rint((centers_by_port[:, 0] - gx[0]) / dx).astype(int)[:, None] + half
    rows = np.rint((centers_by_port[:, 1] - gy[0]) / dy).astype(int)[:, None] + half
    qx = np.broadcast_to(gx[cols][:, None, :], (len(port_xy), window, window))
    qy = np.broadcast_to(gy[rows][:, :, None], (len(port_xy), window, window))
    interp = LinearNDInterpolator(mesh.points[:, :2], intensity.T)
    pix = np.nan_to_num(interp(qx.ravel(), qy.ravel()), nan=0.0)     # (n_px, n)
    signals = pix.reshape(len(port_xy), window * window, -1).mean(axis=1).T

    return LanternDataset(
        signals=signals, coeff=np.asarray(coeff, float), n_train=int(n_train),
        core_xy=port_xy, wavelengths_nm=np.asarray([p["wavelength_nm"]], float),
        meta={"measurement": f"camera: mean of {window}x{window} px windows, "
                             f"{grid_resolution} px grid, unit coupled power",
              "detected_core_xy": np.array2string(centers_by_port, separator=",")})


def dataset_centers(ds) -> np.ndarray:
    """The core centres a camera dataset was measured with (its meta)."""
    import ast
    return np.array(ast.literal_eval(ds.meta["detected_core_xy"]), float)


def simulate_spectra(base_params, ifunc, coeff, n_train, wavelengths_nm,
                     **pipeline_kwargs) -> LanternDataset:
    """Spectral dataset: ``coeff`` (n, n_modes) [nm] -> per-channel coupled
    power fractions (n, n_wl, n_cores), one lantern characterization per
    wavelength (MultiWavelengthPropagationPipeline; ``pipeline_kwargs`` are
    passed to it, e.g. verbose=False)."""
    import multi_wvl as mw

    mwp = mw.MultiWavelengthPropagationPipeline(
        base_params, ifunc, wavelengths_nm, **pipeline_kwargs)
    with contextlib.chdir(CBEAM_EXAMPLES):         # its data/ holds the caches
        power, wl = mwp.get_power_spectra(coeff)                    # (n_wl, n, n_cores)
    return LanternDataset(
        signals=np.ascontiguousarray(power.transpose(1, 0, 2)),
        coeff=np.asarray(coeff, float), n_train=int(n_train),
        core_xy=output_core_xy(base_params), wavelengths_nm=np.asarray(wl, float),
        meta={"measurement": "channel power / incident power, per wavelength"})


def make_single_dataset(n, n_train, span_nm, n_modes=5, rng=0, **kwargs) -> LanternDataset:
    """The single-wavelength dataset of the notebooks: ``n`` random
    configurations of DM modes 0..n_modes-1, uniform in +-span_nm (seeded by
    ``rng``), imaged at 800 nm (simulate_core_images)."""
    p = bp.get_simulation_parameters(3, 0.8)
    ifunc = bp.load_ifunc(p["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    coeff, _ = bp.create_random_aberration_configs(n, n_modes, -span_nm, span_nm, rng=rng)
    return simulate_core_images(p, ifunc, coeff, n_train, **kwargs)


def make_spectral_dataset(n, n_train, span_nm, wavelengths_nm, n_modes=5, rng=0,
                          **pipeline_kwargs) -> LanternDataset:
    """The spectral dataset of the notebooks: like make_single_dataset, but the
    channel powers at every wavelength (simulate_spectra; hours for 50000
    samples x 41 wavelengths -- see data/gen/ for a chunked, parallel run)."""
    base = bp.get_simulation_parameters(nrings=3)
    ifunc = bp.load_ifunc(base["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    coeff, _ = bp.create_random_aberration_configs(n, n_modes, -span_nm, span_nm, rng=rng)
    pipeline_kwargs.setdefault("verbose", False)
    return simulate_spectra(base, ifunc, coeff, n_train, wavelengths_nm, **pipeline_kwargs)


def ramp_spectra(results_file, mode, n_steps, span_nm, wavelengths_nm, rerun=False,
                 prop_chunk_size=32):
    """Channel power spectra of a ramp of one DM mode, n_steps amplitudes from
    -span_nm to +span_nm, at every wavelength (multi_wvl's
    MultiWavelengthPropagationPipeline), cached in ``results_file``.

    Returns ``(power_spectra (n_wl, n_steps, n_cores), wavelengths_nm, coeff, labels)``.
    """
    coeff, labels = bp.create_ramp_aberration_configs([mode], n_steps, -span_nm, span_nm)
    if os.path.exists(results_file) and not rerun:
        with np.load(results_file) as f:
            print(f"loaded {results_file}")
            return f["power_spectra"], f["wavelengths"], coeff, labels
    import multi_wvl as mw
    base = bp.get_simulation_parameters(nrings=3)
    ifunc = bp.load_ifunc(base["ifunc_file"]); bp.center_ifunc_mask(ifunc)
    mwp = mw.MultiWavelengthPropagationPipeline(base, ifunc, wavelengths_nm, verbose=False)
    with contextlib.chdir(CBEAM_EXAMPLES):         # its data/ holds the characterizations
        power, wl = mwp.get_power_spectra(coeff, prop_chunk_size=prop_chunk_size)
    np.savez(results_file, power_spectra=power, wavelengths=wl, coeff_matrix=coeff, labels=labels)
    print(f"saved {results_file}")
    return power, wl, coeff, labels
