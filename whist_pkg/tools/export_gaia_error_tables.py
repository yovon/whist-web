"""Build-time tool: export the Gaia observational-error lookup tables.

The desktop GUI's "Gaia errors (blur HR)" option uses GaiaCatalogErrorModel,
which reads a ~340 MB Gaia FITS to tabulate median sigma(parallax/G/BP/RP) vs
apparent G. The browser can't ship or read that catalog, but the *result* is
tiny — four ~60-point curves over magnitude-bin centers. This tool builds those
curves offline and saves them to catalog/gaia_error_tables.npz, which the web
PrecomputedGaiaErrorModel loads with numpy alone (no FITS, no astropy at runtime).

Usage:
    python export_gaia_error_tables.py [path_to_wd_with_completeness.fits]
"""
import os
import sys

import numpy as np

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../whist_pkg
sys.path.insert(0, os.path.join(PKG, "sim_core"))

from error_models import GaiaCatalogErrorModel

if len(sys.argv) < 2:
    sys.exit("usage: python export_gaia_error_tables.py <path_to_wd_with_completeness.fits>")
src = sys.argv[1]
out = os.path.join(PKG, "catalog", "gaia_error_tables.npz")

model = GaiaCatalogErrorModel(fits_path=src)
model.build()                      # reads the FITS, builds interp1d tables
t = model._tables

# interp1d stores its knots as .x (bin centers) and .y (median sigma per bin).
centers = np.asarray(t["plx"].x, dtype=np.float32)
data = {k: np.asarray(t[k].y, dtype=np.float32) for k in ("plx", "G", "BP", "RP")}

os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez(out, centers=centers, **data)

print(f"source     : {src}")
print(f"n_bins     : {centers.size}")
print(f"G range    : {centers.min():.2f} .. {centers.max():.2f} (apparent G)")
print(f"saved      : {os.path.normpath(out)}  ({os.path.getsize(out)/1024:.1f} KB)")
