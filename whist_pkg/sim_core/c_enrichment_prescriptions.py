"""c_enrichment_prescriptions.py

C-enrichment prescriptions for He-rich white dwarfs (Camisassa et al. 2023,
A&A 674, A213). Implements the four [C/He](Teff) sequences used to model
"stealth DQ" white dwarfs that explain the Gaia CMD bifurcation B branch.

Reference
---------
Camisassa, M. E. et al. (2023). A hidden population of white dwarfs with
atmospheric carbon traces in the Gaia bifurcation. A&A, 674, A213.
https://doi.org/10.1051/0004-6361/202346628
"""

import os
import re

import numpy as np


class StealthDQGrid:
    """
    Bivariate interpolator over the StealthDQ atmospheric model grids
    (Camisassa et al. 2023).  PCHIP over the sparse mass axis, linear over
    the dense Teff axis.  Requires scipy >= 1.10.

    Parameters
    ----------
    grid_root : str
        Path to the StealthDQ directory containing Cseq/, Cseq-1/, Cseq-2/,
        Cseq-3/ sub-folders.
    """

    _SEQ_FOLDERS = {
        'c_sequence': 'Cseq',
        'minus1':     'Cseq-1',
        'minus2':     'Cseq-2',
        'minus3':     'Cseq-3',
    }
    _COL_TEFF, _COL_TCOOL, _COL_G, _COL_BP, _COL_RP = 0, 2, 5, 6, 7
    _N_TEFF = 600   # common Teff grid resolution after resampling
    _N_AGE  = 600   # common cooling-age grid resolution

    def __init__(self, grid_root):
        self._interps      = {}    # seq_key -> 2D RGI (mass, teff) -> color  [legacy]
        self._interp3d     = None  # 3D RGI (mass, teff, offset) -> color     [legacy]
        # Age-based interpolators (mass, age_cool) -> (logteff, G, bprp). Used when the
        # stealth-DQ tracks themselves drive the cooling, so we never compute teff first.
        self._age_interps  = {}    # seq_key -> {'logteff': RGI, 'Mag': RGI, 'color': RGI}
        self._age_interp3d = {}    # {'logteff': RGI, 'Mag': RGI, 'color': RGI} on (mass, age, offset)
        self._load(grid_root)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_mass(prefix):
        """Convert a filename numeric prefix to M☉.  2-digit prefixes like
        '09' are treated as tenths (0.9 M☉); 3-digit as hundredths (0.51 M☉)."""
        v = int(prefix)
        return v / 10.0 if len(prefix) == 2 and v < 20 else v / 100.0

    def _load(self, root):
        from scipy.interpolate import (
            RegularGridInterpolator, LinearNDInterpolator, interp1d)

        seqs     = list(self._SEQ_FOLDERS.keys())  # fixed order: c_sequence … minus3
        seq_data = {}  # seq_key -> per-sequence cache (see below)

        for seq_key, folder in self._SEQ_FOLDERS.items():
            folder_path = os.path.join(root, folder)
            mass_data = {}      # mass -> sorted-by-teff (teff, bprp) for legacy 2D
            scat_mass, scat_age, scat_lt, scat_G, scat_bp = [], [], [], [], []

            for fname in os.listdir(folder_path):
                if not fname.endswith('.dat'):
                    continue
                m = re.match(r'^(\d+)', fname)
                if not m:
                    continue
                mass = self._parse_mass(m.group(1))
                arr  = np.genfromtxt(os.path.join(folder_path, fname), skip_header=1)

                # Legacy view sorted by Teff (powers (mass, teff) -> color).
                order_t = np.argsort(arr[:, self._COL_TEFF])
                mass_data[mass] = (
                    arr[order_t, self._COL_TEFF],
                    arr[order_t, self._COL_BP] - arr[order_t, self._COL_RP],
                )

                # Age-anchored scatter for the (mass, age_cool) interpolators.
                ages = arr[:, self._COL_TCOOL] - np.nanmin(arr[:, self._COL_TCOOL])
                valid = ~(np.isnan(ages) | np.isnan(arr[:, self._COL_TEFF]))
                scat_mass.append(np.full(int(valid.sum()), mass))
                scat_age .append(ages[valid])
                scat_lt  .append(np.log10(arr[valid, self._COL_TEFF]))
                scat_G   .append(arr[valid, self._COL_G])
                scat_bp  .append(arr[valid, self._COL_BP] - arr[valid, self._COL_RP])

            masses = np.array(sorted(mass_data))

            # ---- Legacy 2D (mass, teff) -> color ----
            teff_min = max(mass_data[m][0][0]  for m in masses)
            teff_max = min(mass_data[m][0][-1] for m in masses)
            teff_common = np.linspace(teff_min, teff_max, self._N_TEFF)
            color_grid = np.empty((len(masses), self._N_TEFF))
            for i, mm in enumerate(masses):
                t, c = mass_data[mm]
                color_grid[i] = interp1d(
                    t, c, kind='linear', bounds_error=False, fill_value=np.nan
                )(teff_common)
            self._interps[seq_key] = RegularGridInterpolator(
                (masses, teff_common), color_grid,
                method='pchip', bounds_error=False, fill_value=np.nan,
            )

            # ---- New (mass, age_cool) -> (logteff, G, bprp) ----
            # LinearND scatter -> RegularGrid (matches the camisassa loader pattern), so
            # non-rectangular coverage in (mass, age) doesn't truncate the population to
            # the most-restrictive mass track. The 1.29 M☉ track stops at ~2.4 Gyr but
            # most lower-mass tracks reach 5-8 Gyr; the convex hull captures that.
            mass_s = np.concatenate(scat_mass)
            age_s  = np.concatenate(scat_age)
            lt_s   = np.concatenate(scat_lt)
            G_s    = np.concatenate(scat_G)
            bp_s   = np.concatenate(scat_bp)
            age_lo, age_hi = float(age_s.min()), float(age_s.max())
            m_grid   = np.linspace(masses[0], masses[-1], len(masses) * 20)
            a_grid   = np.linspace(age_lo,    age_hi,     self._N_AGE)
            mm_, aa_ = np.meshgrid(m_grid, a_grid, indexing='ij')
            pts2d    = np.column_stack([mass_s, age_s])
            lt_grid  = LinearNDInterpolator(pts2d, lt_s, rescale=True)(mm_, aa_)
            G_grid   = LinearNDInterpolator(pts2d, G_s,  rescale=True)(mm_, aa_)
            bp_grid  = LinearNDInterpolator(pts2d, bp_s, rescale=True)(mm_, aa_)
            self._age_interps[seq_key] = {
                'logteff': RegularGridInterpolator((m_grid, a_grid), lt_grid,
                                                   method='linear', bounds_error=False, fill_value=np.nan),
                'Mag':     RegularGridInterpolator((m_grid, a_grid), G_grid,
                                                   method='linear', bounds_error=False, fill_value=np.nan),
                'color':   RegularGridInterpolator((m_grid, a_grid), bp_grid,
                                                   method='linear', bounds_error=False, fill_value=np.nan),
            }

            seq_data[seq_key] = (masses, teff_common, color_grid,
                                 m_grid, a_grid, lt_grid, G_grid, bp_grid,
                                 age_lo, age_hi)

        # ---- Legacy 3D (mass, teff, offset) -> color ----
        all_masses  = seq_data[seqs[0]][0]
        teff_lo     = max(sd[1][0]  for sd in seq_data.values())
        teff_hi     = min(sd[1][-1] for sd in seq_data.values())
        teff_3d     = np.linspace(teff_lo, teff_hi, self._N_TEFF)
        offset_axis = np.array([0.0, 1.0, 2.0, 3.0])

        color_3d = np.empty((len(all_masses), self._N_TEFF, 4))
        for j, seq_key in enumerate(seqs):
            _, teff_j, cg_j, *_ = seq_data[seq_key]
            for i in range(len(all_masses)):
                color_3d[i, :, j] = interp1d(
                    teff_j, cg_j[i], kind='linear', bounds_error=False, fill_value=np.nan
                )(teff_3d)

        self._interp3d = RegularGridInterpolator(
            (all_masses, teff_3d, offset_axis), color_3d,
            method='linear', bounds_error=False, fill_value=np.nan,
        )

        # ---- New 3D (mass, age_cool, offset) -> (logteff, G, bprp) ----
        # Build a common (mass, age) grid spanning the union of per-sequence extents,
        # then query each sequence's 2D RGI to populate the offset slice. RGIs return
        # NaN outside the convex hull, which is the desired OOB behaviour.
        m_lo = min(sd[3][0]  for sd in seq_data.values())
        m_hi = max(sd[3][-1] for sd in seq_data.values())
        a_lo = min(sd[8]      for sd in seq_data.values())
        a_hi = max(sd[9]      for sd in seq_data.values())
        m_3d   = np.linspace(m_lo, m_hi, len(all_masses) * 20)
        a_3d   = np.linspace(a_lo, a_hi, self._N_AGE)
        mm3, aa3 = np.meshgrid(m_3d, a_3d, indexing='ij')
        pts_q  = np.column_stack([mm3.ravel(), aa3.ravel()])
        lt_3d  = np.empty((len(m_3d), len(a_3d), 4))
        G_3d   = np.empty((len(m_3d), len(a_3d), 4))
        bp_3d  = np.empty((len(m_3d), len(a_3d), 4))
        for j, seq_key in enumerate(seqs):
            interps = self._age_interps[seq_key]
            lt_3d[:, :, j] = interps['logteff'](pts_q).reshape(mm3.shape)
            G_3d [:, :, j] = interps['Mag']    (pts_q).reshape(mm3.shape)
            bp_3d[:, :, j] = interps['color']  (pts_q).reshape(mm3.shape)
        self._age_interp3d = {
            'logteff': RegularGridInterpolator((m_3d, a_3d, offset_axis), lt_3d,
                                               method='linear', bounds_error=False, fill_value=np.nan),
            'Mag':     RegularGridInterpolator((m_3d, a_3d, offset_axis), G_3d,
                                               method='linear', bounds_error=False, fill_value=np.nan),
            'color':   RegularGridInterpolator((m_3d, a_3d, offset_axis), bp_3d,
                                               method='linear', bounds_error=False, fill_value=np.nan),
        }

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_bprp(self, mass, teff, sequence):
        """
        Interpolate Bp3-Rp3 color for a named C-enrichment sequence.

        Parameters
        ----------
        mass : array-like  (M☉)
        teff : array-like  (K)
        sequence : {'c_sequence', 'minus1', 'minus2', 'minus3'}

        Returns
        -------
        ndarray — intrinsic Bp3-Rp3.  NaN where outside grid bounds.
        """
        mass = np.asarray(mass, dtype=float).ravel()
        teff = np.asarray(teff, dtype=float).ravel()
        return self._interps[sequence](np.column_stack([mass, teff]))

    def get_bprp_offset(self, mass, teff, offset):
        """
        Interpolate color for a continuous dex offset in [0, 3].

        offset=0 → c_sequence, 1 → minus1, 2 → minus2, 3 → minus3.
        Uses a pre-built 3D RegularGridInterpolator (mass, teff, offset) so all
        stars are handled in a single vectorized call.

        Parameters
        ----------
        mass, teff : array-like, shape (N,)
        offset : array-like, shape (N,)  per-star dex offsets

        Returns
        -------
        ndarray shape (N,) — Bp3-Rp3.  NaN outside grid bounds.
        """
        mass   = np.asarray(mass,   dtype=float).ravel()
        teff   = np.asarray(teff,   dtype=float).ravel()
        offset = np.clip(np.asarray(offset, dtype=float).ravel(), 0.0, 3.0)
        return self._interp3d(np.column_stack([mass, teff, offset]))

    def get_observables(self, mass, age_cool, sequence):
        """Vectorized (logteff, G, bprp) lookup on (mass, age_cool) for a named sequence.

        Lets callers skip the cooling-track teff step entirely when the stealth-DQ
        atmosphere itself is driving the cooling — the .trk-style files for each
        C sequence carry Teff, G3, and Bp3-Rp3 as functions of cooling age.

        Returns
        -------
        (logteff, G, bprp) : tuple of ndarrays — NaN outside grid bounds.
        """
        mass = np.asarray(mass, dtype=float).ravel()
        age  = np.asarray(age_cool, dtype=float).ravel()
        pts  = np.column_stack([mass, age])
        fns  = self._age_interps[sequence]
        return fns['logteff'](pts), fns['Mag'](pts), fns['color'](pts)

    def get_observables_offset(self, mass, age_cool, offset):
        """Vectorized (logteff, G, bprp) lookup on (mass, age_cool, offset) for a
        continuous dex offset in [0, 3]."""
        mass   = np.asarray(mass,   dtype=float).ravel()
        age    = np.asarray(age_cool, dtype=float).ravel()
        offset = np.clip(np.asarray(offset, dtype=float).ravel(), 0.0, 3.0)
        pts    = np.column_stack([mass, age, offset])
        d3 = self._age_interp3d
        return d3['logteff'](pts), d3['Mag'](pts), d3['color'](pts)


class CEnrichmentModel:
    """
    Parametric [C/He] vs. Teff prescriptions from Camisassa et al. (2023).

    All four prescriptions are offsets of the observed DQ C-sequence:
      - c_sequence:      offset =  0 dex  (optically detectable C)
      - c_sequence_minus1: offset = -1 dex  (near optical detection limit)
      - c_sequence_minus2: offset = -2 dex  (stealth DQ, main class)
      - c_sequence_minus3: offset = -3 dex  (thick He envelope, faint C)

    Temperature regions
    -------------------
    Teff > 12000 K  : pure He, [C/He] = PURE_HE_LIMIT = -10.41
    Teff < 10000 K  : linear interpolation through reference points
    10000 <= Teff <= 12000 K : reflected about 10000 K
    """

    PURE_HE_LIMIT  = -10.41   # [C/He] threshold for treating as pure He
    TEFF_TRANSITION = 12000.0  # K — above this, pure He
    TEFF_PIVOT      = 10000.0  # K — reflection point for warm region

    # Reference points: (Teff K, [C/He] dex) for the base C sequence
    _TEFF_REF = np.array([8000.0, 9000.0, 10000.0])
    _CHE_REF  = np.array([-5.20,  -5.85,  -6.50])

    # Approximate peak Bp-Rp shift (mag) at C-sequence level; negative = bluer.
    # Parametric stand-in until full Koester He/C atmosphere grids are loaded.
    _MAX_DELTA_BPRP = -0.15

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _base_c_sequence(self, teff):
        """
        Compute base [C/He] (offset = 0) as a numpy array for any teff input.
        Scalar inputs are broadcast to 0-d arrays and returned as such.
        """
        teff = np.asarray(teff, dtype=float)
        scalar = teff.ndim == 0
        teff = np.atleast_1d(teff)

        result = np.full_like(teff, self.PURE_HE_LIMIT)

        # Cool region: Teff <= 10000 K (np.interp clamps at edges)
        cool = teff < self.TEFF_PIVOT
        if cool.any():
            result[cool] = np.interp(teff[cool], self._TEFF_REF, self._CHE_REF)

        # Warm region: 10000 K <= Teff <= 12000 K — reflect about 10000 K
        warm = (teff >= self.TEFF_PIVOT) & (teff <= self.TEFF_TRANSITION)
        if warm.any():
            teff_refl = 20000.0 - teff[warm]
            result[warm] = np.interp(teff_refl, self._TEFF_REF, self._CHE_REF)

        # Hot region: Teff > 12000 K → already set to PURE_HE_LIMIT

        return result.squeeze() if scalar else result

    def _apply_offset(self, teff, offset):
        base = self._base_c_sequence(teff)
        return np.maximum(base - offset, self.PURE_HE_LIMIT)

    # ---------------------------------------------------------------------------
    # Public prescription methods
    # ---------------------------------------------------------------------------

    def c_sequence(self, teff):
        """C sequence (offset = 0): optically detectable C (true DQ class)."""
        return self._apply_offset(teff, 0.0)

    def c_sequence_minus1(self, teff):
        """C sequence −1 dex: near optical detection limit."""
        return self._apply_offset(teff, 1.0)

    def c_sequence_minus2(self, teff):
        """C sequence −2 dex: stealth DQ (main class), clearly sub-optical."""
        return self._apply_offset(teff, 2.0)

    def c_sequence_minus3(self, teff):
        """C sequence −3 dex: thick He envelope, faint C."""
        return self._apply_offset(teff, 3.0)

    def c_random(self, teff, random_range=(0.0, 2.0), rng=None):
        """
        Assign a random dex offset per star, uniformly drawn from random_range.

        Parameters
        ----------
        teff : array-like
            Effective temperatures (K).
        random_range : tuple(float, float)
            (min, max) of the uniform offset distribution (dex).
        rng : numpy.random.Generator, optional
            Random number generator. If None, uses numpy default.

        Returns
        -------
        c_he : ndarray
            [C/He] values enforced >= PURE_HE_LIMIT.
        offsets : ndarray
            Per-star dex offsets that were drawn.
        """
        if rng is None:
            rng = np.random.default_rng()
        teff = np.asarray(teff, dtype=float)
        offsets = rng.uniform(random_range[0], random_range[1], size=teff.shape)
        base = self._base_c_sequence(teff)
        return np.maximum(base - offsets, self.PURE_HE_LIMIT), offsets

    def get_c_he(self, teff, prescription='minus2', rng=None):
        """
        Dispatch to the requested prescription by name.

        Parameters
        ----------
        teff : array-like
            Effective temperatures (K).
        prescription : str
            One of: 'c_sequence', 'minus1', 'minus2', 'minus3',
            'random_0_2', 'random_0_3'  (random_<min>_<max> format).
        rng : numpy.random.Generator, optional
            Used only for random prescriptions.

        Returns
        -------
        ndarray of [C/He] values.
        """
        dispatch = {
            'c_sequence': self.c_sequence,
            'minus1':     self.c_sequence_minus1,
            'minus2':     self.c_sequence_minus2,
            'minus3':     self.c_sequence_minus3,
        }
        if prescription in dispatch:
            return dispatch[prescription](teff)
        if prescription.startswith('random'):
            parts = prescription.split('_')   # e.g. ['random', '0', '2']
            lo = float(parts[1]) if len(parts) > 1 else 0.0
            hi = float(parts[2]) if len(parts) > 2 else 2.0
            result, _ = self.c_random(teff, random_range=(lo, hi), rng=rng)
            return result
        raise ValueError(
            f"Unknown prescription '{prescription}'. "
            "Valid options: c_sequence, minus1, minus2, minus3, random_<lo>_<hi>."
        )

    # ---------------------------------------------------------------------------
    # Approximate photometric correction
    # ---------------------------------------------------------------------------

    def approx_delta_bprp(self, c_he, teff):
        """
        Approximate Gaia Bp-Rp color correction from C enrichment.

        Parametric stand-in for full Koester He/C atmosphere grids (Camisassa
        et al. 2023). Trace C provides free electrons → enhanced He⁻ free-free
        opacity → bluer continuum → negative (bluer) Bp-Rp shift.

        The temperature dependence is implicit: [C/He] = PURE_HE_LIMIT for
        Teff > 12000 K, so the effect vanishes automatically above the
        transition temperature.

        Parameters
        ----------
        c_he : array-like
            [C/He] abundance (dex), as returned by the prescription methods.
        teff : array-like
            Effective temperatures (K) — used for range gating only.

        Returns
        -------
        delta : ndarray
            Δ(Bp-Rp) in magnitudes. Negative means bluer.
            Replace this method with a grid interpolation when Koester He/C
            models are available.
        """
        c_he = np.asarray(c_he, dtype=float)
        teff = np.asarray(teff, dtype=float)

        # Fractional C-enrichment effect: 0 at pure He, 1 at C-sequence peak
        ref_c = self._CHE_REF[0]   # -5.2 dex (C-sequence at 8000 K)
        effect = np.clip(
            (c_he - self.PURE_HE_LIMIT) / (ref_c - self.PURE_HE_LIMIT),
            0.0, 1.0
        )

        return self._MAX_DELTA_BPRP * effect
