"""Per-lantern settings of the notebooks.

Piston ranges (uniform, each of the 5 petals), the same rule for every lantern:
+-lambda/2 at one wavelength (single_span_nm) and +-8 lambda for the 41-wavelength
dispersed datasets (spectral_span_nm), lambda = the design wavelength.

The lantern is named by CBEAM_LANTERN (one of cbeam's batch_pipeline.LANTERNS:
pl19_800, the reference 19-port lantern at 800 nm, by default; pl7_1200;
pl19_1200). Everything a notebook would otherwise hard-code for one lantern --
the single-wavelength piston range, the polynomial budgets, the Zernike basis
size, where its datasets and models live -- comes from here, so the same
notebook runs for every lantern; its executed copy goes to runs/<lantern>/.
"""
from __future__ import annotations

import os

_CUBIC_19 = dict(  # 19 core signals: 1539 degree-3 monomials
    omp_ks=[100, 200, 300, 400, 500, 600, 750, 900, 1050, 1150, 1282],
    omp_record=(1282,),            # trainMLP: the budget chosen from the sweep
    omp_record_wb=(500, 1282),     # workbench Part A
    k_table=500, ks_mix=[100, 200, 300, 400, 500], k_importance=1282,
    n_zern=36, zern_spectral=18,
    k_wrap=1000,                   # per-wavelength OMP budget (of 1539 terms)
    wl_stride_global=16,           # global OMP: 3 wavelengths x 19 cores = 57 inputs
)

_SPANS_800 = dict(single_span_nm=400.0, spectral_span_nm=6400.0)     # lambda = 800 nm
_SPANS_1200 = dict(single_span_nm=600.0, spectral_span_nm=9600.0)   # lambda = 1200 nm

ML = {
    "pl19_800": dict(_CUBIC_19, **_SPANS_800),
    "pl19_1200": dict(_CUBIC_19, **_SPANS_1200),
    "pl7_1200": dict(  # 7 core signals: 119 degree-3 monomials in all
        **_SPANS_1200,
        omp_ks=[5, 10, 20, 30, 40, 60, 80, 100, 119],
        omp_record=(119,), omp_record_wb=(40, 119),
        k_table=60, ks_mix=[5, 10, 20, 30, 60], k_importance=119,
        n_zern=10, zern_spectral=10,   # Z1-Z10: the fewest with rank 7 on the hexagon
        k_wrap=119,
        wl_stride_global=8,            # 6 wavelengths x 7 cores = 42 inputs
    ),
}


def lantern() -> str:
    """The lantern the notebooks run for (CBEAM_LANTERN, default pl19_800)."""
    name = os.environ.get("CBEAM_LANTERN", "pl19_800")
    if name not in ML:
        raise ValueError(f"CBEAM_LANTERN={name!r}: known lanterns are {sorted(ML)}")
    return name


def settings(name: str | None = None) -> dict:
    """ML settings of a lantern, plus its data paths: ``data_dir`` (data/ml for
    the reference lantern, its original location; data/ml/<lantern> else),
    ``single`` and ``spectral`` (the two datasets) and ``ramp`` (the
    multi_wavelength ramp results)."""
    name = name or lantern()
    s = dict(ML[name], lantern=name)
    d = "data/ml" if name == "pl19_800" else f"data/ml/{name}"
    s.update(data_dir=d,
             single=f"{d}/single_800nm.npz" if name == "pl19_800" else f"{d}/single.npz",
             spectral=f"{d}/spectral_41wl.npz",
             ramp=f"{d}/mw_ramp_41.npz")
    return s
