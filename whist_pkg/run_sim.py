"""run_sim.py — STEP 1 verification.

Proves the self-contained sim package computes one default frame end-to-end
using ONLY the KEEP dependencies (numpy/scipy/pandas/astropy/matplotlib), with
no reference to the original repo and no pip-installed WD_models. This is the
exact compute path the browser build will call.

Run:
    python run_sim.py
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "sim_core"))  # top-level sim modules

import matplotlib
matplotlib.use("Agg")  # headless; never needs a display/Qt
import numpy as np

import simulate_wd_full_population as sim  # noqa: F401  (import-chain check)
from simulate_wd_full_population import (
    DefaultParameters, create_population, extract_observables,
)
import WD_models_loader_wrapper


def run_once(p, loader):
    pop = create_population(p, show_times=False)
    pop, mask = extract_observables(
        pop, p, loader=loader, show_timings=False, use_direct_approach=True,
    )
    return pop, mask


def main():
    p = dict(DefaultParameters.defaults)
    loader = WD_models_loader_wrapper.WDModelsLoader(
        HR_grid=p["HR_grid"], HR_bands=p["HR_bands"],
        lazy=True, interp_method="grid", grid_res=(300, 800),
    )

    t0 = time.perf_counter()
    pop, mask = run_once(p, loader)            # cold: builds grids once
    t_cold = time.perf_counter() - t0

    t0 = time.perf_counter()
    pop2, mask2 = run_once(p, loader)          # warm: grids cached
    t_warm = time.perf_counter() - t0

    overlay = np.load(os.path.join(HERE, "catalog", "gaia_hr_overlay.npy"))

    print("\n" + "=" * 70)
    print("STEP 1 — self-contained package ran end-to-end")
    print("=" * 70)
    print(f"extract+create (cold) : {t_cold*1000:8.1f} ms  (one-time grid build)")
    print(f"extract+create (warm) : {t_warm*1000:8.1f} ms  (per parameter change)")
    print(f"population            : {pop.shape[0]} rows x {pop.shape[1]} cols")
    print(f"selected             : {int(mask.sum())} objects")
    print(f"plot arrays          : bprp{pop['bprp'].values.shape}, "
          f"G{pop['G'].values.shape}, true_age{pop['true_age'].values.shape}")
    print(f"gaia overlay (.npy)  : {overlay.shape} dtype={overlay.dtype} "
          f"(cols: bp_rp, absG)")

    # Prove no CUT subsystem was pulled in by the package.
    CUT = ("PySide6", "PyQt6", "PyQt5", "shiboken", "emcee", "corner",
           "astroquery", "requests", "imageio", "streamlit", "plotly", "tqdm")
    leaked = sorted({m.split(".")[0] for m in sys.modules
                     if m.split(".")[0] in CUT})
    print("-" * 70)
    print(f"CUT deps imported    : {leaked if leaked else 'NONE'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
