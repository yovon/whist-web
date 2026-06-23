import itertools
from collections import defaultdict
import sys, os
import numpy as np
from pathlib import Path

BASE_DIR  = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
# Locate the vendored "White Dwarf Models" dir (holds the WD_models package +
# cooling grids). In this package it sits one level up from sim_core/; the
# original repo relied on WD_models being pip-installed, so tolerate both.
for _cand in (BASE_DIR / "White Dwarf Models", BASE_DIR.parent / "White Dwarf Models"):
    if _cand.exists():
        sys.path.insert(0, str(_cand))
        break
import WD_models as WD_models

class WDModelsLoader:
    """
    Manage loading/caching of WD_models.load_model results and building 2D interpolators.
    Usage:
      loader = WDModelsLoader(HR_grid=..., HR_bands=..., lazy=True)
      loader.build_cache()                # optional (lazy default)
      tpl = loader.choose_tuple(atm='He', thickness='thin', core='CO', mass=0.95)
      teff_fn = loader.get_interpolator(tpl, output_key='logteff')
      logteff = teff_fn(mass_array, age_array)
    """

    # Short-name aliases accepted by WD_models.load_model; needed here because we
    # call read_cooling_tracks directly and must pass the expanded names.
    _ALIAS_MAP = {
        'be': 'Bedard2020', 'bet': 'Bedard2020_thin',
        'f':  'Fontaine2001', 'ft': 'Fontaine2001_thin',
        'b':  'BaSTI',  'bn': 'BaSTI_nosep',
        'c':  'Camisassa2017',
        'o':  'ONe',
        'm':  'MESA',
        'r001':  'Renedo2010_001',
        'r0001': 'Renedo2010_0001',
    }

    _DISTIL_MASSES   = [1.00, 1.05, 1.10, 1.15, 1.20, 1.25]
    _DISTIL_FILENAME = "{mass_int:d}_605030_DSTL2.txt"  # mass_int = int(m * 100) #NOTE: this is the same pattern used in the added notebook to Bedard2024
    _DISTIL_DIR_DEFAULT = None  # resolved lazily relative to this file

    _DISTIL_ATM_DA_PATTERN  = "Table_Mass_{mass:.2f}_DA"
    _DISTIL_ATM_DQ_PATTERN  = "Table_Mass_{mass:.2f}_DQ_Cm1p0"
    # Column indices in STELUM atm tables (same for both DA and DQ, 0-based):
    _DISTIL_ATM_COL_G3    = 37
    _DISTIL_ATM_COL_G3_BP = 38
    _DISTIL_ATM_COL_G3_RP = 39

    def __init__(self, model_tuples=None, HR_grid=None, HR_bands=None, lazy=True,
                 interp_method='grid', grid_res=(200, 800), distil_dir=None):
        """
        Parameters
        ----------
        interp_method : {'grid', 'delaunay'}
            'grid'     — RegularGridInterpolator on a precomputed regular (mass × y) grid
                         (default). Delaunay is used once at build time to fill the grid;
                         subsequent queries are O(N_q). Negligibly less accurate near boundaries.
            'delaunay' — LinearNDInterpolator on scatter data (Delaunay triangulation).
                         Accurate everywhere; query cost O(N_q log N_g).
        grid_res : (int, int)
            (n_mass, n_y) resolution of the regular grid used when interp_method='grid'.
        distil_dir : str or None
            Path to the STELUM_sequences directory containing distillation tracks.
            If None, defaults to <repo_root>/White Dwarf Models/STELUM_sequences.
        """
        self.HR_grid = HR_grid
        self.HR_bands = HR_bands if HR_bands is not None else ('bp3-rp3', 'G3')
        self.lazy = lazy
        self.interp_method = interp_method
        self.grid_res = grid_res
        self.distil_dir = distil_dir
        self.model_tuples = model_tuples if model_tuples is not None else self.default_tuples()
        # caches
        self.model_cache = {t: None for t in self.model_tuples} if lazy else {}
        self.interp_cache = {}  # keys: (tuple, output_key, y_axis, interp_method)
        self._distil_interp = None  # built lazily on first get_distil_teff call
        self._distil_atm_interp = {}  # atm_type_lower -> (mag_fn, color_fn), built by build_distil_atm_cache()
        self.sdq_grid = None       # StealthDQGrid, built by build_sdq_cache()
        self.camisassa_grid = None # CamisassaCoolingGrid, built by build_camisassa_cache()
        # shared objects reused across tuples with the same atm_type
        self._atm_fn_cache = {}    # (atm_type, band_str) -> callable (logteff, logg) -> value
        self._logg_func_cache = {} # atm_type -> logg_func for BaSTI models
        if not lazy:
            self.build_cache()

        # normalized arrays for quick selection
        self._sync_tuple_arrays()

    @staticmethod
    def default_tuples():
        return [
            ("be", "be", "be", "H"),
            ("bet", "bet", "bet", "H"),
            ("be", "be", "o", "H"),
            ("be", "be", "be", "He"),
            # ("bet", "bet", "bet", "He"),
            ("be", "c", "o", "He")
        ]

    def _sync_tuple_arrays(self):
        self._a0 = np.array([t[0].lower() for t in self.model_tuples])
        self._a1 = np.array([t[1].lower() for t in self.model_tuples])
        self._a2 = np.array([t[2].lower() for t in self.model_tuples])
        self._atm = np.array([t[3].lower() for t in self.model_tuples])

    def build_cache(self):
        """Load all models into memory using the fast path (no HR-grid building)."""
        for t in self.model_tuples:
            self._load_if_needed(t)
        self._sync_tuple_arrays()

    def _get_atm_fn(self, atm_type, band):
        """Return a cached (logteff, logg) -> photometry callable, building it if needed."""
        key = (atm_type, band)
        if key not in self._atm_fn_cache:
            try:
                _, fn = WD_models.interp_atm(atm_type, band)
            except Exception:
                fn = None
            self._atm_fn_cache[key] = fn
        return self._atm_fn_cache[key]

    def _load_if_needed(self, tpl):
        """Fast model loader: bypasses load_model to skip the 7 interp_HR_to_para calls.

        Calls read_cooling_tracks + interp_atm directly, caching atmosphere functions
        and logg_func by atm_type so they are shared across all tuples that use the
        same atmosphere composition.
        """
        mdl = self.model_cache.get(tpl)
        if mdl is not None:
            return mdl

        low, mid, high, atm_type = tpl
        # expand short aliases to the full names expected by read_cooling_tracks
        low  = self._ALIAS_MAP.get(low,  low)
        mid  = self._ALIAS_MAP.get(mid,  mid)
        high = self._ALIAS_MAP.get(high, high)

        try:
            # get (or build+cache) atmosphere interpolation functions
            to_color = self._get_atm_fn(atm_type, self.HR_bands[0])
            to_BC    = self._get_atm_fn(atm_type, self.HR_bands[1] + '-Mbol')

            # logg_func is only needed when BaSTI tracks are included
            logg_func = None
            if 'BaSTI' in mid or 'BaSTI' in high:
                if atm_type not in self._logg_func_cache:
                    m_b, lg_b, _, _, lt_b, _ = WD_models.read_cooling_tracks(
                        'Bedard2020', 'Bedard2020', 'Bedard2020', atm_type)
                    self._logg_func_cache[atm_type] = WD_models.interp_xy_z_func(
                        lt_b, m_b, lg_b)
                logg_func = self._logg_func_cache[atm_type]

            mass_array, logg, age, age_cool, logteff, Mbol = \
                WD_models.read_cooling_tracks(low, mid, high, atm_type, logg_func)

            if to_BC is not None and to_color is not None:
                Mag   = to_BC(logteff, logg) + Mbol
                color = to_color(logteff, logg)
            else:
                Mag   = np.full_like(logteff, np.nan)
                color = np.full_like(logteff, np.nan)

            mdl = {
                'mass_array': mass_array,
                'logg':       logg,
                'age':        age,
                'age_cool':   age_cool,
                'logteff':    logteff,
                'Mbol':       Mbol,
                'Mag':        Mag,
                'color':      color,
            }
        except Exception as e:
            print(f"Warning: failed loading {tpl}: {e}")
            mdl = None

        self.model_cache[tpl] = mdl
        return mdl

    def get_model(self, tpl):
        """Return loaded model dict (load if lazy)."""
        return self._load_if_needed(tpl)

    def get_hr_contour_model(self, tpl, hr_grid=None):
        """Return a minimal dict with grid_HR_to_mass and grid_HR_to_age for contour plots.

        Builds only the 2 grids actually used by plots.py, reusing the already-loaded
        model data.  Uses a coarser step than the full HR grid so griddata is fast
        (matplotlib interpolates contour lines regardless of grid density).
        """
        from scipy.interpolate import griddata as _griddata

        mdl = self._load_if_needed(tpl)
        if mdl is None:
            return {}

        color      = mdl['color']
        Mag        = mdl['Mag']
        mass_array = mdl['mass_array']
        age        = mdl['age']

        # default: coarse grid — same range as hr_grid but 5× coarser steps
        if hr_grid is None:
            hr_grid = (-0.5, 2.0, 0.01, 9.0, 16.0, 0.05)
        cmin, cmax, dc, mmin, mmax, dm = hr_grid
        gx, gy = np.mgrid[cmin:cmax:dc, mmin:mmax:dm]

        sel = (~np.isnan(color + Mag) &
               (Mag > mmin) & (Mag < mmax) &
               (color > cmin) & (color < cmax))
        pts = np.column_stack([color[sel], Mag[sel]])

        grid_mass = _griddata(pts, mass_array[sel], (gx, gy),
                              method='linear', rescale=True)
        grid_age  = _griddata(pts, age[sel],        (gx, gy),
                              method='linear', rescale=True)

        return {'grid_HR_to_mass': grid_mass, 'grid_HR_to_age': grid_age}

    def get_interpolator(self, tpl, output_key='logteff', y_axis=None):
        """
        Return a 2D interpolator function f(mass, y) -> output_key.

        Which backend is used is controlled by self.interp_method:
          'delaunay' — LinearNDInterpolator (Delaunay triangulation on scatter data)
          'grid'     — RegularGridInterpolator on a precomputed regular (mass × y) grid
        Both have the same call signature: fn(mass_array, y_array) -> ndarray.
        """
        cache_key = (tpl, output_key, y_axis, self.interp_method)
        if cache_key in self.interp_cache:
            return self.interp_cache[cache_key]

        model = self._load_if_needed(tpl)
        if model is None or output_key not in model:
            self.interp_cache[cache_key] = None
            return None

        # determine y-axis array
        y_arr = None
        if y_axis is not None:
            ya = str(y_axis)
            if ya in model:
                y_arr = model[ya]
            elif ya == 'age_cool' and 'age_cool' in model:
                y_arr = model['age_cool']
            elif ya == 'logteff' and 'logteff' in model:
                y_arr = model['logteff']
        else:
            if output_key.lower() in ('mag', 'color'):
                if 'logteff' in model:
                    y_arr = model['logteff']
                elif 'age_cool' in model:
                    y_arr = model['age_cool']
            else:
                if 'age_cool' in model:
                    y_arr = model['age_cool']
                elif 'logteff' in model:
                    y_arr = model['logteff']

        if y_arr is None:
            self.interp_cache[cache_key] = None
            return None

        x_arr = model.get('mass_array')
        z_arr = model[output_key]

        try:
            if self.interp_method == 'grid':
                fn = self._build_grid_interp(x_arr, y_arr, z_arr)
            else:
                fn = WD_models.interp_xy_z_func(x=x_arr, y=y_arr, z=z_arr, interp_type='linear')
        except Exception:
            fn = None

        self.interp_cache[cache_key] = fn
        return fn

    def _build_grid_interp(self, x, y, z):
        """Build a RegularGridInterpolator over a regular (x × y) grid.

        Uses LinearNDInterpolator (Delaunay) once to fill the grid, then discards
        the triangulation. All subsequent queries use O(N_q) array lookups.
        """
        from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

        n_x, n_y = self.grid_res
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[valid], y[valid], z[valid]

        x_grid = np.linspace(x.min(), x.max(), n_x)
        y_grid = np.linspace(y.min(), y.max(), n_y)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing='ij')

        tri = LinearNDInterpolator(np.column_stack([x, y]), z, rescale=True)
        zz  = tri(xx, yy)  # shape (n_x, n_y); NaN outside convex hull

        rgi = RegularGridInterpolator(
            (x_grid, y_grid), zz,
            method='linear', bounds_error=False, fill_value=np.nan,
        )
        return lambda mass, age: rgi(np.column_stack([
            np.asarray(mass, dtype=float).ravel(),
            np.asarray(age,  dtype=float).ravel(),
        ]))

    def _build_distil_interp(self):
        """Read STELUM distillation tracks once and build an interpolator.

        Uses self.interp_method:
          'delaunay' — LinearNDInterpolator on the scatter points
          'grid'     — RegularGridInterpolator on a precomputed regular (mass × age) grid
        """
        import pandas as pd
        from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator

        stelum_dir = self.distil_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'White Dwarf Models', 'STELUM_sequences',
        )

        age_list, mass_list, teff_list = [], [], []
        for m in self._DISTIL_MASSES:
            fname = self._DISTIL_FILENAME.format(mass_int=int(round(m * 100)))
            path  = os.path.join(stelum_dir, fname)
            tbl   = pd.read_csv(path, skiprows=9, sep=r'\s+',
                                 names=['age', 'teff', 'logg', 'radius', 'luminosity'])
            ages = tbl['age'].values
            age_list.append(ages)
            mass_list.append(np.full(len(ages), m))
            teff_list.append(tbl['teff'].values)

        ages_arr   = np.concatenate(age_list)
        masses_arr = np.concatenate(mass_list)
        teff_arr   = np.concatenate(teff_list)

        if self.interp_method == 'grid':
            n_mass, n_age = self.grid_res
            mass_grid = np.linspace(masses_arr.min(), masses_arr.max(), n_mass)
            age_grid  = np.linspace(ages_arr.min(),   ages_arr.max(),   n_age)
            mm, aa    = np.meshgrid(mass_grid, age_grid, indexing='ij')
            tri       = LinearNDInterpolator(np.column_stack([ages_arr, masses_arr]), teff_arr, rescale=True)
            zz        = tri(np.column_stack([aa.ravel(), mm.ravel()])).reshape(n_mass, n_age)
            rgi       = RegularGridInterpolator(
                (mass_grid, age_grid), zz,
                method='linear', bounds_error=False, fill_value=np.nan,
            )
            self._distil_interp = lambda m, a: rgi(np.column_stack([
                np.asarray(m, dtype=float).ravel(),
                np.asarray(a, dtype=float).ravel(),
            ]))
        else:
            tri = LinearNDInterpolator(np.column_stack([ages_arr, masses_arr]), teff_arr, rescale=True)
            self._distil_interp = lambda m, a: tri(np.column_stack([
                np.asarray(a, dtype=float).ravel(),
                np.asarray(m, dtype=float).ravel(),
            ]))

    def build_distil_cache(self):
        """Pre-build and cache the distillation-track interpolator (STELUM sequences).

        Called explicitly during startup so the first extract_observables call
        does not pay the file I/O + triangulation cost.
        """
        if self._distil_interp is None:
            self._build_distil_interp()

    def build_sdq_cache(self, grid_root):
        """Load and cache the StealthDQ photometry grid (Camisassa et al. 2023).

        Called once at startup so extract_observables never rebuilds it.
        Subsequent calls with a different grid_root replace the cached grid.

        Parameters
        ----------
        grid_root : str
            Path to the StealthDQ directory containing Cseq/, Cseq-1/, Cseq-2/, Cseq-3/.
        """
        if self.sdq_grid is None or getattr(self.sdq_grid, '_grid_root', None) != grid_root:
            from c_enrichment_prescriptions import StealthDQGrid
            self.sdq_grid = StealthDQGrid(grid_root)
            self.sdq_grid._grid_root = grid_root  # store for identity check above

    def build_camisassa_cache(self, grid_root):
        """Load and cache the Camisassa H/He cooling+photometry grid.

        Tracks bundle LPCODE-consistent Gaia EDR3 photometry directly, so the
        Montreal/Bergeron atmosphere step is bypassed for stars routed through
        this grid. Called once at startup; subsequent calls with the same root
        are no-ops.

        Parameters
        ----------
        grid_root : str
            Path to the Camisassa directory containing DA/ and DB/ subfolders.
        """
        if self.camisassa_grid is None or getattr(self.camisassa_grid, '_grid_root', None) != grid_root:
            from camisassa_cooling import CamisassaCoolingGrid
            self.camisassa_grid = CamisassaCoolingGrid(grid_root, grid_res=self.grid_res)
            self.camisassa_grid._grid_root = grid_root

    def get_distil_teff(self, masses, ages):
        """Interpolate distillation-track Teff for given (masses, ages).

        Builds the interpolator from STELUM CSV files on first call and caches it.
        All subsequent calls use the cached interpolator with no file I/O.
        """
        if self._distil_interp is None:
            self._build_distil_interp()
        return self._distil_interp(masses, ages)

    def _build_distil_atm_interp(self, atm_type):
        """Read STELUM atmospheric tables and build (mass, logteff) -> photometry interpolators.

        Builds two callables stored in self._distil_atm_interp[atm_type.lower()]:
          mag_fn(masses, logteff)   -> G3 absolute magnitude
          color_fn(masses, logteff) -> G3_BP - G3_RP color
        """
        import pandas as pd

        atm_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'White Dwarf Models', 'STELLUM atm',
        )
        is_da = atm_type.lower() in ('h', 'da')
        pattern = self._DISTIL_ATM_DA_PATTERN if is_da else self._DISTIL_ATM_DQ_PATTERN

        mass_list, logteff_list, g3_list, bprp_list = [], [], [], []
        for m in self._DISTIL_MASSES:
            fname = pattern.format(mass=m)
            path  = os.path.join(atm_dir, fname)
            tbl   = pd.read_csv(path, skiprows=2, sep=r'\s+', header=None, na_values=['NaN', 'nan'])
            teff_vals = tbl.iloc[:, 0].values.astype(float)
            g3_vals   = tbl.iloc[:, self._DISTIL_ATM_COL_G3].values.astype(float)
            bprp_vals = (tbl.iloc[:, self._DISTIL_ATM_COL_G3_BP].values.astype(float) -
                         tbl.iloc[:, self._DISTIL_ATM_COL_G3_RP].values.astype(float))
            logteff_vals = np.log10(teff_vals)
            valid = ~(np.isnan(g3_vals) | np.isnan(bprp_vals) | np.isnan(logteff_vals))
            mass_list.append(np.full(valid.sum(), m))
            logteff_list.append(logteff_vals[valid])
            g3_list.append(g3_vals[valid])
            bprp_list.append(bprp_vals[valid])

        masses_arr  = np.concatenate(mass_list)
        logteff_arr = np.concatenate(logteff_list)
        g3_arr      = np.concatenate(g3_list)
        bprp_arr    = np.concatenate(bprp_list)

        self._distil_atm_interp[atm_type.lower()] = (
            self._build_grid_interp(masses_arr, logteff_arr, g3_arr),
            self._build_grid_interp(masses_arr, logteff_arr, bprp_arr),
        )

    def build_distil_atm_cache(self):
        """Pre-build and cache the STELUM atmospheric interpolators for both H and He."""
        for atm_type in ('H', 'He'):
            if atm_type.lower() not in self._distil_atm_interp:
                self._build_distil_atm_interp(atm_type)

    def get_distil_photometry(self, masses, logteff, atm_type):
        """Return (G, bprp) for distilled stars using the STELUM atmospheric tables.

        Parameters
        ----------
        masses, logteff : array-like
            Stellar masses [Msun] and log10(Teff) values.
        atm_type : str
            Atmosphere type: 'H' (DA) or 'He' (DQ).
        """
        key = atm_type.lower()
        if key not in self._distil_atm_interp:
            self._build_distil_atm_interp(atm_type)
        mag_fn, color_fn = self._distil_atm_interp[key]
        return mag_fn(masses, logteff), color_fn(masses, logteff)

    def choose_tuple(self, atm, thickness, core, mass_is_midrange):
        """
        Choose the best tuple from self.model_tuples for given star properties.
        Rules implemented:
          - match atmosphere (H/He) first
          - if core is Ne-like prefer tuples with 'o' in third arg
          - if atm=='He' and mass in mid range prefer tuples containing 'c'
          - prefer thickness ('bet' for thin, 'be' for thick) if available
          - fallback to first matching-atm tuple or global first tuple
        Returns: tuple (the load_model args)
        """
        atm_l = str(atm).lower()
        thick_l = str(thickness).lower()
        core_l = None if core is None else str(core).lower()

        # candidate indices matching atmosphere (fallback to all)
        cand = np.where(self._atm == atm_l)[0]
        if cand.size == 0:
            cand = np.arange(len(self.model_tuples))

        # prefer Ne-core -> 'o' in 3rd arg
        if core_l is not None and core_l.startswith('ne'):
            cand_o = [ci for ci in cand if self._a2[ci] == 'o' or 'o' in (self._a0[ci], self._a1[ci], self._a2[ci])]
            if cand_o:
                cand = np.array(cand_o)

        # prefer 'c' for He mid-range
        if atm_l == 'he' and mass_is_midrange:
            cand_c = [ci for ci in cand if 'c' in (self._a0[ci], self._a1[ci], self._a2[ci])]
            if cand_c:
                cand = np.array(cand_c)

        # thickness preference
        prefer = 'bet' if ('thin' in thick_l or 'bet' in thick_l) else 'be'
        for ci in cand:
            if prefer in (self._a0[ci], self._a1[ci], self._a2[ci]):
                return self.model_tuples[int(ci)]

        # fallback to first candidate
        return self.model_tuples[int(cand[0])]