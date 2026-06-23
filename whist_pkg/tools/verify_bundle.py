"""verify_bundle.py — extract web/whist_bundle.zip to a temp dir and run the
compute core against the EXTRACTED layout, exactly as Pyodide does after
unpackArchive. Catches bundle path/nesting bugs before opening the browser.

Run:  python verify_bundle.py
"""
import os
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # WHIST-Webapp
ZIP = os.path.join(ROOT, "web", "whist_bundle.zip")

tmp = tempfile.mkdtemp(prefix="whist_verify_")
dest = os.path.join(tmp, "whist_pkg")          # mimics Pyodide's /whist_pkg
with zipfile.ZipFile(ZIP) as z:
    z.extractall(dest)

sys.path.insert(0, os.path.join(dest, "sim_core"))

import matplotlib
matplotlib.use("Agg")
import numpy as np
from simulate_wd_full_population import (
    DefaultParameters, create_population, extract_observables,
)
import WD_models_loader_wrapper

p = dict(DefaultParameters.defaults)
loader = WD_models_loader_wrapper.WDModelsLoader(
    HR_grid=p["HR_grid"], HR_bands=p["HR_bands"],
    lazy=True, interp_method="grid", grid_res=(300, 800),
)
t0 = time.perf_counter()
pop = create_population(p, show_times=False)
pop, mask = extract_observables(pop, p, loader=loader, show_timings=False,
                                use_direct_approach=True)
dt = time.perf_counter() - t0
overlay = np.load(os.path.join(dest, "catalog", "gaia_hr_overlay.npy"))

print(f"extracted to : {dest}")
print(f"compute (cold): {dt*1000:.0f} ms")
print(f"population    : {pop.shape}, selected: {int(np.sum(np.asarray(mask)))}, "
      f"overlay: {overlay.shape}")
print("BUNDLE OK — unpacked layout resolves and the compute core runs from it.")
