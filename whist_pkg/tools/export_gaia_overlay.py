"""Build-time tool: export the default Gaia HR overlay to a static .npy file.

The web demo shows the observed Gaia white-dwarf HR diagram behind the simulated
population. The compute core needs no catalog data; this overlay is the *only*
catalog touch, and it happens OFFLINE at build time — never in the browser.

It reads the local Gaia WD catalog, applies the desktop GUI's default 120 pc
distance filter (1000/parallax < 120), and saves (bp_rp, absG) as a float32
(N, 2) array. Output loads in Pyodide with numpy alone (np.load), no pyarrow.

Usage:
    python export_gaia_overlay.py [path_to_gaia_wd_local.fits]
"""
import os
import sys

import numpy as np
from astropy.table import Table

DISTANCE_PC = 120.0  # GUI default (gui_settings.json): 1000/parallax < distance_pc

if len(sys.argv) < 2:
    sys.exit("usage: python export_gaia_overlay.py <path_to_gaia_wd_local.fits>")
src = sys.argv[1]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "..", "catalog", "gaia_hr_overlay.npy")

t = Table.read(src, hdu=1)  # already normalised to parallax / bp_rp / absG
parallax = np.asarray(t["parallax"], float)
bp_rp = np.asarray(t["bp_rp"], float)
absG = np.asarray(t["absG"], float)

finite = np.isfinite(parallax) & np.isfinite(bp_rp) & np.isfinite(absG)
keep = finite & (parallax > 1000.0 / DISTANCE_PC)  # distance < 120 pc

arr = np.column_stack([bp_rp[keep], absG[keep]]).astype(np.float32)
os.makedirs(os.path.dirname(out), exist_ok=True)
np.save(out, arr)

print(f"source                : {src}")
print(f"source rows           : {len(t)}")
print(f"within {DISTANCE_PC:g} pc        : {int(keep.sum())}")
print(f"saved                 : {os.path.normpath(out)}")
print(f"  shape={arr.shape} dtype={arr.dtype}  size={os.path.getsize(out)/1024:.1f} KB")
print("columns: [:,0]=bp_rp  [:,1]=absG")
