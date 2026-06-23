# whist_pkg — self-contained WD population simulation core

The minimal simulation compute core behind [WHIST](../README.md): the Qt GUI,
MCMC fitting, catalogue downloads, and video recorder of the original
population-synthesis project are removed. It computes one HR-diagram frame from
parameters using only the dependencies available under Pyodide, so it runs both
in a plain Python env and in the browser.

## Layout
```
whist_pkg/
  sim_core/                 # top-level sim modules (added to sys.path)
    simulate_wd_full_population.py   # engine; tqdm dependency removed
    WD_models_loader_wrapper.py      # grid loader; WD_models discovery generalised
    selection_models.py
    error_models.py                  # default error_model=None (no catalog read)
    c_enrichment_prescriptions.py
    camisassa_cooling.py
  White Dwarf Models/       # vendored model grids (was pip-installed / repo-root)
    WD_models/              # the external interpolation package + cooling_models/
    Camisassa/              # default cooling_source grids (DA/, DB/)
    Camisassa StealthDQ/    # C-enrichment grids (Cseq*)
    STELUM_sequences/       # distillation tracks
    STELLUM atm/            # distilled-star photometry tables
  catalog/
    gaia_hr_overlay.npy     # pre-exported observed overlay: float32 (N,2) = (bp_rp, absG)
  tools/
    export_gaia_overlay.py  # build-time: regenerate the overlay from the local Gaia FITS
  run_sim.py                # verification: compute one default frame, print shapes
  requirements-web.txt      # KEEP deps (Pyodide-available)
```

## Run (plain Python env, KEEP deps only)
```
python -m venv .venv-web
.venv-web/Scripts/python -m pip install numpy scipy pandas astropy matplotlib
python run_sim.py
```
Expected: ~20.9k objects created / ~12.7k selected, warm compute ~65 ms,
`CUT deps imported: NONE`.

## Compute entry point
The browser build should call, directly (not `simulate_HR_all`, which also
plots/saves):
```python
import simulate_wd_full_population as sim
from simulate_wd_full_population import DefaultParameters, create_population, extract_observables
import WD_models_loader_wrapper

p = dict(DefaultParameters.defaults)
loader = WD_models_loader_wrapper.WDModelsLoader(HR_grid=p["HR_grid"], HR_bands=p["HR_bands"],
                                                 lazy=True, interp_method="grid", grid_res=(300,800))
pop = create_population(p, show_times=False)
pop, mask = extract_observables(pop, p, loader=loader, show_timings=False, use_direct_approach=True)
# plot arrays: pop["bprp"], pop["G"] (absolute), pop["true_age"]; mask selects observed objects
```
Build the loader **once** and reuse it across runs — the ~6 s first-call grid
build is then paid only at startup; subsequent runs are ~65 ms.

## What was changed vs. the source repo
- `simulate_wd_full_population.py`: the `tqdm` dependency is removed — the two
  progress-bar wrappers now iterate the plain group lists directly (progress is
  still reported through `progress_callback`).
- `WD_models_loader_wrapper.py`: the `White Dwarf Models` lookup now also checks
  the parent dir, so the vendored package is found without a pip install.
- Everything else is a byte-for-byte copy.

## Notes
- Bundle is ~44 MB raw model grids; trimmable for the web (STELUM uses 6 of 78
  files; WD_models cooling_models can drop BaSTI/Fontaine/Renedo/MESA — the
  default tuples use only Bedard2020/ONe/Camisassa2017).
- The overlay is the default 120 pc Gaia sample (`1000/parallax < 120`).
- Model-grid attribution and references: see [Credits](../README.md#credits--white-dwarf-models).
