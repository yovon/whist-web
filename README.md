# WHIST — White-dwarf HR-diagram Interactive Simulation Tool

An interactive, **in-browser** white-dwarf population simulator. Adjust the
parameters of a Galactic white-dwarf population model and watch the simulated
Gaia colour–magnitude (HR) diagram update in real time, overlaid on the observed
120 pc Gaia white-dwarf sample.

Everything runs **client-side** in the browser via [Pyodide](https://pyodide.org/)
— the scientific Python stack (NumPy/SciPy/pandas/Astropy/Matplotlib) and the
simulation core are compiled to WebAssembly. **There is no backend**; once the
page loads, all computation happens on your machine.

## Features

- Full parameter panel (IMF, star-formation history, IFMR, binary fraction,
  carbon-enrichment / "stealth DQ" prescriptions, 22Ne distillation probability,
  Gaia selection, …).
- Four views: HR density, catalogue comparison, residual, and parameter
  distributions.
- Gaia error blurring from pre-exported σ-tables, and absolute/apparent-magnitude
  selection.
- Interactive plot overlay: box/wheel zoom, pan, home, and polygon selection that
  drives the distribution panel.

## Run it locally

Pyodide must fetch the simulation bundle over `http://` (browsers block `file://`),
so serve the `web/` folder rather than double-clicking the HTML.

```bash
cd web
python serve.py            # or: python -m http.server 8000
```

Then open <http://localhost:8000/>.

The first load takes ~30–90 s while the browser downloads Pyodide and the
scientific stack and builds the cooling grids once. After that, each parameter
change recomputes a frame in a fraction of a second.

## Repository layout

```
WHIST-Webapp/
  web/
    index.html            the app (Pyodide + UI + plotting)
    serve.py              tiny static server for local use
    whist_bundle.zip      sim_core + model grids + Gaia overlay, zipped for the browser
  whist_pkg/              the self-contained simulation package (see whist_pkg/README.md)
    sim_core/             the compute modules
    White Dwarf Models/   vendored cooling/atmosphere model grids (see Credits)
    catalog/              pre-exported Gaia HR overlay + error σ-tables
    tools/                build/export/verify scripts
    run_sim.py            verify one frame in a plain Python env
```

To verify the compute core outside the browser:

```bash
python -m venv .venv-web
.venv-web/Scripts/python -m pip install numpy scipy pandas astropy matplotlib
.venv-web/Scripts/python whist_pkg/run_sim.py
```

## Credits — White Dwarf Models

This tool is a thin interactive wrapper around published white-dwarf cooling
tracks, atmosphere models, and the interpolation tooling that ties them together.
**All the underlying science belongs to the authors below.** The model grids in
`whist_pkg/White Dwarf Models/` are redistributed for convenience and remain
subject to their original authors' terms; please cite the corresponding papers if
you use this work.

### `WD_models` — interpolation package
The HR-diagram ↔ white-dwarf-parameter interpolation framework, by **Sihao Cheng**
(Institute for Advanced Study). See `whist_pkg/White Dwarf Models/WD_models/README.md`
for full usage and the model list.
- Package & docs: <https://github.com/SihaoCheng/WD_models>
- Default model tuple used here: **Bédard et al. (2020)** cooling tracks with the
  **Montreal/Bergeron** synthetic atmospheres, plus the **ONe** ultramassive tracks.

### Camisassa CO white-dwarf cooling tracks (`Camisassa/`)
La Plata **LPCODE** evolutionary sequences with LPCODE-consistent photometry.
Contact / tables: maria.camisassa@upc.edu (Z = 0.02; DA & DB; 1 Oct 2024 update).
- Althaus et al. (2013), A&A 557, A19 (He-core)
- Camisassa et al. (2016), ApJ 823, 158 (CO-core)
- Camisassa et al. (2017), ApJ 839, 11 (He-atmosphere CO)
- Camisassa et al. (2019), A&A 625, A87 (ultramassive ONe-core)
- Atmospheres: Koester (2010), MmSAI 81, 921

### "Stealth DQ" carbon-enrichment grids (`Camisassa StealthDQ/`)
Atmospheric [C/He](Teff) sequences modelling the carbon-bearing white dwarfs that
explain the Gaia colour–magnitude bifurcation (B branch).
- **Camisassa, M. E. et al. (2023)**, *A hidden population of white dwarfs with
  atmospheric carbon traces in the Gaia bifurcation*, A&A 674, A213.
  <https://doi.org/10.1051/0004-6361/202346628>

### STELUM distillation sequences & atmospheres (`STELUM_sequences/`, `STELLUM atm/`)
Tracks computed with the Montreal **STELUM** stellar-evolution code, including
22Ne-distillation sequences and the corresponding atmosphere tables.
- STELUM code: **Bédard, Bergeron, Brassard, Fontaine (2022)**, ApJ 927, 128.
- Distillation: **Bédard, Blouin & Cheng (2024)**, *Buoyant crystals halt the
  cooling of white dwarf stars*, Nature 627, 286.

### Observational overlay
The HR overlay is the **Gaia** 120 pc white-dwarf sample (selected via
`1000/parallax < 120`), pre-exported to `whist_pkg/catalog/gaia_hr_overlay.npy`.
- **Gentile Fusillo, N. P. et al. (2021)** — Gaia EDR3 white-dwarf catalogue, *MNRAS* 508, 3877.

### References

If you use this tool or the bundled model grids, please cite the relevant works:

- Althaus, L. G. et al. (2013), *A&A* 557, A19.
- Bédard, A., Bergeron, P., Brassard, P. & Fontaine, G. (2020), *ApJ* 901, 93.
- Bédard, A., Bergeron, P., Brassard, P. & Fontaine, G. (2022), *ApJ* 927, 128.
- Bédard, A., Blouin, S. & Cheng, S. (2024), *Buoyant crystals halt the cooling of white dwarf stars*, *Nature* 627, 286.
- Bergeron, P. et al. (2011), *ApJ* 737, 28.
- Camisassa, M. E. et al. (2016), *ApJ* 823, 158.
- Camisassa, M. E. et al. (2017), *ApJ* 839, 11.
- Camisassa, M. E. et al. (2019), *A&A* 625, A87.
- Camisassa, M. E. et al. (2023), *A hidden population of white dwarfs with atmospheric carbon traces in the Gaia bifurcation*, *A&A* 674, A213. https://doi.org/10.1051/0004-6361/202346628
- Cheng, S., *WD_models* — white-dwarf model interpolation package. https://github.com/SihaoCheng/WD_models
- Gentile Fusillo, N. P. et al. (2021) — Gaia EDR3 white-dwarf catalogue, *MNRAS* 508, 3877.
- Koester, D. (2010), *MmSAI* 81, 921.

---

*The references above are drawn from the citations embedded in the model files
and source code. If any DOI, volume, or attribution needs correcting, please
open an issue.*
