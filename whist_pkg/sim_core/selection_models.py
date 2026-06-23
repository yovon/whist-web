"""selection_models.py

Modular observational selection / volume-sampling layer.

Subclasses of `SelectionModel` decide which simulated WDs pass the
"observed" filter for a given observational setup (volume-limited cube,
Gaia magnitude limit, empirical S(M_G, BP-RP), etc.).

Important invariant: **subclasses never modify the absolute photometry**
columns. `all_data["G"]` and `all_data["bprp"]` remain absolute magnitudes
and the canonical absG column. Distance-aware subclasses may compute an
apparent-magnitude scratch column internally for filtering, but the plot
layer continues to read absG from `all_data["G"]`.

The default `AbsoluteMagCap` reproduces the legacy behaviour exactly:
keep stars with `absG <= params["G_max"]`, no distance sampling.
"""

from __future__ import annotations

import numpy as np


class SelectionModel:
    """Base class for observational selection."""

    #: Distance-aware models set this True so `extract_observables` knows it must
    #: draw the per-star distance (`_sel_distance`, pc) *before* calling `apply`.
    needs_distance = False

    def apply(self, all_data, params, rng=None):
        """Return a boolean mask of length `len(all_data)` of accepted stars.

        Implementations must not mutate `all_data["G"]` or `all_data["bprp"]`.
        Distance-aware models may add scratch columns prefixed with `_sel_`
        for internal bookkeeping; the plot layer ignores those.
        """
        raise NotImplementedError


class AbsoluteMagCap(SelectionModel):
    """Hard absolute-magnitude cap. Reproduces the pre-modular behaviour.

    Selection: `absG <= G_max`. Distance is implicit (10 pc, so absG == G).
    Reads `G_max` from params, falling back to the constructor value.
    """

    def __init__(self, G_max: float = np.inf):
        self.G_max = float(G_max)

    def apply(self, all_data, params, rng=None):
        G_max = float(params.get("G_max", self.G_max))
        return all_data["G"].values <= G_max


class ApparentMagCap(SelectionModel):
    """Volume-limited, *apparent*-magnitude cap — the physically correct Gaia
    selection once distances exist.

    Selection: `G_app <= G_max`, where `G_app = absG + 5*log10(d/10)` and the
    per-star distance `d` (pc) is the `_sel_distance` column drawn by
    `extract_observables` (uniform-in-volume within `volume_radius_pc`; see
    docs/programmatic/gaia_observational_errors.md for why uniform).

    Reads `G_max` from params, falling back to the constructor value. Does not
    mutate `G`/`bprp`; it only reads the `_sel_distance` scratch column.
    """

    needs_distance = True

    def __init__(self, G_max: float = np.inf):
        self.G_max = float(G_max)

    def apply(self, all_data, params, rng=None):
        G_max = float(params.get("G_max", self.G_max))
        if "_sel_distance" not in all_data.columns:
            raise RuntimeError(
                "ApparentMagCap requires the '_sel_distance' column; "
                "extract_observables must draw distances before selection."
            )
        d = all_data["_sel_distance"].values
        g_app = all_data["G"].values + 5.0 * np.log10(d) - 5.0
        return g_app <= G_max
