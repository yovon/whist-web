"""build_web_bundle.py — package the sim core + grids + overlay into one zip
that the Pyodide PoC fetches and unpacks into the browser filesystem.

Output: WHIST-Webapp/web/whist_bundle.zip with top-level entries
  sim_core/...   "White Dwarf Models"/...   catalog/...
The PoC unpacks it into /whist_pkg in the Pyodide FS, so that each module's
`__file__/../White Dwarf Models` resolves exactly as in a local run.

Run:  python build_web_bundle.py
"""
import os
import zipfile

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../whist_pkg
ROOT = os.path.dirname(PKG)                                        # .../WHIST-Webapp
OUT = os.path.join(ROOT, "web", "whist_bundle.zip")

INCLUDE = ["sim_core", "White Dwarf Models", "catalog"]
EXCLUDE_DIRS = {"__pycache__"}
EXCLUDE_EXT = {".pyc", ".pyo"}

os.makedirs(os.path.dirname(OUT), exist_ok=True)

n = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for top in INCLUDE:
        base = os.path.join(PKG, top)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
            for f in filenames:
                if os.path.splitext(f)[1].lower() in EXCLUDE_EXT:
                    continue
                full = os.path.join(dirpath, f)
                arc = os.path.relpath(full, PKG).replace(os.sep, "/")  # zip uses /
                z.write(full, arc)
                n += 1

print(f"wrote   : {OUT}")
print(f"files   : {n}")
print(f"zip size: {os.path.getsize(OUT) / 1e6:.1f} MB")
