"""camisassa_cooling.py

Self-contained loader for the Camisassa H- and He-atmosphere CO white dwarf
cooling tracks (Camisassa et al. 2016, 2017, plus 2024 updates), bundled with
LPCODE-consistent Gaia EDR3 photometry inside the same .trk files.

This sits alongside `StealthDQGrid` and the wrapper's STELUM distillation
interpolators: it bypasses the WD_models module entirely so the photometry
shipped in the .trk files (LPCODE atmosphere) is preserved rather than being
overridden by the Montreal/Bergeron Delaunay atmosphere fit.

Directory layout expected:
    <root>/DA/0.2Msun.trk ... 1.3Msun.trk     (H atmosphere)
    <root>/DB/0.5Msun.trk ... 1.3Msun.trk     (He atmosphere)

Each .trk file is whitespace-delimited with two header lines starting with '#'.
Column order (0-indexed):
    0 log(Teff)   1 log(L)   2 log(g)   3 age(Gyrs)   4 R(Rsun)   5 Mbol
    6 G3          7 Bp3      8 Rp3       ... photometric bands ...

`age(Gyrs)` is the cooling age (starts near 0 at the hot end of each track and
grows to ~15-16 Gyr at the cool end). We subtract the per-track minimum as a
defensive offset so cool_age=0 maps to the youngest point on each track.

Reference
---------
Camisassa, M. E. et al. (2016). ApJ 823, 158.
Camisassa, M. E. et al. (2017). ApJ 839, 11.
Camisassa, M. E. et al. (2019). A&A 625, A87.
README.txt distributed with the table set (1 Oct 2024 update).
"""

from __future__ import annotations
import os
import re
from pathlib import Path

import numpy as np


class CamisassaCoolingGrid:
    """
    Bivariate (mass, age_cool) -> photometry interpolator built from the
    Camisassa H/He CO white-dwarf cooling tables.

    Builds three RegularGridInterpolators per atmosphere (mag, color, logteff)
    by Delaunay-sampling the (mass, age_cool) scatter onto a regular mesh.
    Mirrors the wrapper's `_build_grid_interp` pattern so accuracy and NaN-
    handling match the Bedard/Bedard+ONe code paths exactly (including
    rescale=True on the Delaunay step).

    Parameters
    ----------
    root_dir : str or Path
        Directory containing DA/ and DB/ subfolders.
    grid_res : tuple(int, int)
        (n_mass, n_age) resolution of the regular grid. Default (200, 800)
        matches the wrapper default.
    """

    # File column indices (0-based)
    _COL_LOGTEFF = 0
    _COL_LOGL    = 1
    _COL_LOGG    = 2
    _COL_AGE     = 3
    _COL_R       = 4
    _COL_MBOL    = 5
    _COL_G3      = 6
    _COL_BP3     = 7
    _COL_RP3     = 8

    # Mass nodes shipped with the Camisassa update (mass in M_sun, file stem)
    _MASSES_DA = ["0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8",
                  "0.9", "1.0", "1.1", "1.2", "1.3"]
    _MASSES_DB = ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"]

    def __init__(self, root_dir, grid_res=(200, 800)):
        self.root_dir = Path(root_dir)
        self.grid_res = grid_res
        # interps[atm_lower] = {'Mag': fn, 'color': fn, 'logteff': fn}
        self._interps = {}
        # raw scatter arrays kept for downstream HR-contour plotting if wanted
        self._scatter = {}
        self._load()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_one_atm(self, atm_letter):
        """Read all .trk files for one atmosphere ('H' or 'He').

        Returns dict of stacked 1-D arrays: mass, age_cool, logteff, Mag, color.
        """
        if atm_letter == "H":
            subdir = self.root_dir / "DA"
            masses = self._MASSES_DA
        elif atm_letter == "He":
            subdir = self.root_dir / "DB"
            masses = self._MASSES_DB
        else:
            raise ValueError(f"atm_letter must be 'H' or 'He', got {atm_letter!r}")

        mass_list, age_list, logteff_list, mag_list, color_list = [], [], [], [], []

        for ms in masses:
            m = float(ms)
            fpath = subdir / f"{ms}Msun.trk"
            if not fpath.exists():
                continue
            data = np.genfromtxt(fpath, comments="#", invalid_raise=False)
            if data.ndim != 2 or data.shape[0] == 0:
                continue

            logteff = data[:, self._COL_LOGTEFF]
            ages    = data[:, self._COL_AGE]
            G3      = data[:, self._COL_G3]
            Bp3     = data[:, self._COL_BP3]
            Rp3     = data[:, self._COL_RP3]
            color   = Bp3 - Rp3

            # Anchor cooling age at 0 for the youngest (hottest) point in the track
            ages_cool = ages - np.nanmin(ages)

            valid = (~np.isnan(logteff) & ~np.isnan(ages_cool) &
                     ~np.isnan(G3) & ~np.isnan(color))
            if not valid.any():
                continue

            mass_list.append(np.full(valid.sum(), m))
            age_list.append(ages_cool[valid])
            logteff_list.append(logteff[valid])
            mag_list.append(G3[valid])
            color_list.append(color[valid])

        if not mass_list:
            raise FileNotFoundError(
                f"No Camisassa {atm_letter} .trk files found under {subdir}")

        return {
            "mass":     np.concatenate(mass_list),
            "age_cool": np.concatenate(age_list),
            "logteff":  np.concatenate(logteff_list),
            "Mag":      np.concatenate(mag_list),
            "color":    np.concatenate(color_list),
        }

    def _load(self):
        for atm in ("H", "He"):
            try:
                scat = self._load_one_atm(atm)
            except FileNotFoundError:
                continue
            self._scatter[atm.lower()] = scat
            self._interps[atm.lower()] = {
                "Mag":     self._build_grid_interp(scat["mass"], scat["age_cool"], scat["Mag"]),
                "color":   self._build_grid_interp(scat["mass"], scat["age_cool"], scat["color"]),
                "logteff": self._build_grid_interp(scat["mass"], scat["age_cool"], scat["logteff"]),
            }

    def _build_grid_interp(self, x, y, z):
        """Mirror of WDModelsLoader._build_grid_interp (with rescale=True)."""
        from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator
        n_x, n_y = self.grid_res
        valid = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
        x, y, z = x[valid], y[valid], z[valid]
        x_grid = np.linspace(x.min(), x.max(), n_x)
        y_grid = np.linspace(y.min(), y.max(), n_y)
        xx, yy = np.meshgrid(x_grid, y_grid, indexing="ij")
        tri = LinearNDInterpolator(np.column_stack([x, y]), z, rescale=True)
        zz  = tri(xx, yy)
        rgi = RegularGridInterpolator(
            (x_grid, y_grid), zz,
            method="linear", bounds_error=False, fill_value=np.nan,
        )
        return lambda mass, age: rgi(np.column_stack([
            np.asarray(mass, dtype=float).ravel(),
            np.asarray(age,  dtype=float).ravel(),
        ]))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def supports(self, atm_type):
        """Whether this grid has tracks for the given atmosphere ('H'/'He')."""
        return str(atm_type).lower() in self._interps

    def mass_range(self, atm_type):
        """(min, max) mass coverage in M_sun for the requested atmosphere."""
        scat = self._scatter[str(atm_type).lower()]
        return float(scat["mass"].min()), float(scat["mass"].max())

    def get_photometry(self, masses, ages_cool, atm_type):
        """Interpolate (G3, bp3-rp3) for stars with given mass and cooling age.

        Parameters
        ----------
        masses, ages_cool : array-like
            Stellar masses (M_sun) and cooling ages (Gyr).
        atm_type : {'H', 'He'}.

        Returns
        -------
        (Mag, color) : tuple of ndarrays
            G3 absolute magnitudes and Bp3-Rp3 colors. NaN outside grid bounds.
        """
        key = str(atm_type).lower()
        fns = self._interps[key]
        return fns["Mag"](masses, ages_cool), fns["color"](masses, ages_cool)

    def get_logteff(self, masses, ages_cool, atm_type):
        """Interpolate log10(Teff) on the (mass, age_cool) plane."""
        key = str(atm_type).lower()
        return self._interps[key]["logteff"](masses, ages_cool)

    def get_all(self, masses, ages_cool, atm_type):
        """Vectorized (G3, bp3-rp3, logTeff) lookup."""
        key = str(atm_type).lower()
        fns = self._interps[key]
        return (fns["Mag"](masses, ages_cool),
                fns["color"](masses, ages_cool),
                fns["logteff"](masses, ages_cool))
