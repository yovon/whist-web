"""error_models.py

Modular Gaia observational-error layer.

Adds Gaia-like measurement uncertainties to the simulated *absolute* photometry
so that synthetic populations can be compared like-for-like against real Gaia
catalogues (which already carry observational scatter). Adding errors to the
simulation, rather than trying to remove them from the data, is the statistically
clean way to match the two samples.

Design mirrors ``selection_models.py``: pluggable ``ErrorModel`` subclasses, an
opt-in ``error_model`` parameter, and a default of ``None`` (no perturbation, so
the pipeline stays byte-identical to the pre-error behaviour).

Invariant
---------
Error models READ the absolute photometry columns (``absG``, ``bprp``) and the
per-star distance scratch column (``_sel_distance``, in pc) produced by the
distance-assignment step in ``extract_observables``. They WRITE separate
*observed* columns (``absG_obs``, ``bprp_obs``, ``parallax_obs``). They never
mutate ``absG``/``bprp``, so every existing reader of the absolute photometry is
unaffected; downstream comparison/plot code opts in to the ``*_obs`` columns
explicitly.

Error relations (data-driven)
-----------------------------
``sigma_parallax``, ``sigma_G``, ``sigma_BP`` and ``sigma_RP`` are tabulated as a
function of *apparent* G directly from a real Gaia catalogue (median error per
magnitude bin) and interpolated. This keeps the noise model self-consistent with
the comparison sample rather than relying on published analytic fits.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.interpolate import interp1d


class ErrorModel:
    """Base class for an observational-error (noise) layer.

    Implementations populate the observed photometry columns
    (``absG_obs``, ``bprp_obs``, ``parallax_obs``) on ``all_data`` in place and
    must not mutate the absolute photometry columns ``absG`` / ``bprp``.
    """

    def apply(self, all_data, params, rng=None):
        raise NotImplementedError


# Repo-root-relative default catalogue used to build the empirical error tables.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
_DEFAULT_CATALOG = os.path.join(_REPO_ROOT, "Catalogues", "wd_with_completeness.fits")


class GaiaCatalogErrorModel(ErrorModel):
    """Data-driven Gaia error model.

    Builds median sigma(parallax), sigma(G), sigma(BP), sigma(RP) vs apparent G
    from a Gaia catalogue, then perturbs the simulated observables:

      * parallax is scattered by sigma_parallax(G_app) and inverted back to a
        perturbed distance, giving a perturbed absolute magnitude;
      * G is scattered by sigma_G(G_app);
      * the colour BP-RP is scattered by sqrt(sigma_BP^2 + sigma_RP^2),
        evaluated at apparent G (a good approximation for white dwarfs, whose
        BP/RP magnitudes track G closely over this colour range).

    The error tables are built lazily on first use and cached on the instance so
    that repeated calls (e.g. MCMC chains) do not re-read the FITS file.
    """

    def __init__(self, fits_path: str | None = None, n_bins: int = 60,
                 g_range: tuple[float, float] | None = None):
        self.fits_path = fits_path or _DEFAULT_CATALOG
        self.n_bins = int(n_bins)
        self.g_range = g_range            # None -> use catalogue min/max apparent G
        self._tables = None               # dict of interp1d, built lazily

    def __deepcopy__(self, memo):
        # The model is effectively immutable once its tables are built (apply()
        # takes the rng as an argument and writes only to the passed DataFrame).
        # Sharing the instance across deepcopies avoids re-reading the catalog —
        # critical for MCMC, where base_params is deep-copied on every eval.
        return self

    def build(self):
        """Eagerly build the error tables (else they build on first apply()).

        Call once before launching many evals (e.g. MCMC) so the catalog read
        happens up front, not inside the first likelihood evaluation.
        """
        if self._tables is None:
            self._build_tables()
        return self

    # ------------------------------------------------------------------ tables
    def _build_tables(self):
        """Read the catalogue once and tabulate median error vs apparent G."""
        from astropy.table import Table

        tbl = Table.read(self.fits_path, hdu=1)
        g = np.asarray(tbl["phot_g_mean_mag"], dtype=float)

        err_cols = {
            "plx": "parallax_error",
            "G":   "phot_g_mean_mag_error",
            "BP":  "phot_bp_mean_mag_error",
            "RP":  "phot_rp_mean_mag_error",
        }

        finite = np.isfinite(g)
        for c in err_cols.values():
            finite &= np.isfinite(np.asarray(tbl[c], dtype=float))
        g = g[finite]

        lo, hi = self.g_range if self.g_range is not None else (np.min(g), np.max(g))
        edges = np.linspace(lo, hi, self.n_bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        which = np.clip(np.digitize(g, edges) - 1, 0, self.n_bins - 1)

        tables = {}
        for key, col in err_cols.items():
            vals = np.asarray(tbl[col], dtype=float)[finite]
            med = np.full(self.n_bins, np.nan)
            for b in range(self.n_bins):
                sel = which == b
                if np.any(sel):
                    med[b] = np.median(vals[sel])
            # fill empty bins by interpolation over populated centers
            ok = np.isfinite(med)
            if not np.all(ok):
                med = np.interp(centers, centers[ok], med[ok])
            # clamp to the populated G range at the edges (flat extrapolation)
            tables[key] = interp1d(centers, med, kind="linear",
                                   bounds_error=False,
                                   fill_value=(med[0], med[-1]))
        self._tables = tables

    # ------------------------------------------------------------------- apply
    def apply(self, all_data, params, rng=None):
        if self._tables is None:
            self._build_tables()
        if rng is None:
            rng = np.random.default_rng(params.get("seed"))

        n = len(all_data)
        absG = np.asarray(all_data["absG"], dtype=float)
        bprp = np.asarray(all_data["bprp"], dtype=float)
        dist = np.asarray(all_data["_sel_distance"], dtype=float)  # pc

        dmod = 5.0 * np.log10(dist) - 5.0           # distance modulus
        g_app = absG + dmod                          # true apparent G
        plx_true = 1000.0 / dist                     # true parallax [mas]

        t = self._tables
        s_plx = t["plx"](g_app)
        s_g = t["G"](g_app)
        s_color = np.hypot(t["BP"](g_app), t["RP"](g_app))

        # perturb
        plx_obs = plx_true + rng.normal(0.0, s_plx, n)
        g_app_obs = g_app + rng.normal(0.0, s_g, n)
        bprp_obs = bprp + rng.normal(0.0, s_color, n)

        # invert the perturbed parallax back to a distance; guard against
        # non-positive draws (faint/low-S/N tail) -> mark observed mag invalid.
        with np.errstate(divide="ignore", invalid="ignore"):
            dist_obs = 1000.0 / plx_obs
            dmod_obs = 5.0 * np.log10(dist_obs) - 5.0
            absG_obs = g_app_obs - dmod_obs
        bad = ~np.isfinite(absG_obs)
        absG_obs[bad] = np.nan
        bprp_obs[bad] = np.nan

        all_data["parallax_obs"] = plx_obs
        all_data["absG_obs"] = absG_obs
        all_data["bprp_obs"] = bprp_obs
        return all_data


class PrecomputedGaiaErrorModel(GaiaCatalogErrorModel):
    """Gaia error model backed by pre-exported sigma tables (web / browser use).

    Identical perturbation math to GaiaCatalogErrorModel, but loads the median
    sigma(parallax/G/BP/RP)-vs-apparent-G curves from a tiny .npz built offline
    by tools/export_gaia_error_tables.py — so no ~340 MB Gaia FITS (and no
    astropy.table) is needed at runtime. Used by the Pyodide web app.
    """

    def __init__(self, npz_path: str):
        super().__init__(fits_path=None)
        self.npz_path = npz_path

    def _build_tables(self):
        d = np.load(self.npz_path)
        centers = np.asarray(d["centers"], dtype=float)
        self._tables = {
            key: interp1d(centers, np.asarray(d[key], dtype=float), kind="linear",
                          bounds_error=False,
                          fill_value=(float(d[key][0]), float(d[key][-1])))
            for key in ("plx", "G", "BP", "RP")
        }
