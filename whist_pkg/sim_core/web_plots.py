"""web_plots.py — matplotlib rendering for the WHIST web app.

Vendored verbatim from the desktop GUI's `gui_modules/plots.py` so the in-browser
HR diagram, catalog-comparison, residual and distribution plots are pixel-for-pixel
the same as the Qt application. Pure matplotlib / numpy / scipy / skimage — no Qt,
so it imports cleanly under Pyodide.
"""

import numpy as np
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap, SymLogNorm
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
# skimage is imported lazily inside _compute_catalog_frangi so this module loads
# even where scikit-image is absent (the Frangi ridge overlay is optional).


def _setup_hr_ax(ax, cmin, cmax, mmin, mmax):
    ax.invert_yaxis()
    ax.set_xlim(cmin - 0.1, cmax + 0.1)
    ax.set_ylim(mmax + 0.1, mmin - 0.1)
    ax.set_xlabel(r"$G_{BP} - G_{RP}$", fontsize=10, color="white")
    ax.set_ylabel(r"$M_G$",             fontsize=10, color="white")
    ax.tick_params(colors="white")


def _hist2d(x, y, bins, cmin, cmax, mmin, mmax, weights=None):
    H, xe, ye = np.histogram2d(x, y, bins=bins,
                                range=[[cmin, cmax], [mmin, mmax]],
                                weights=weights)
    return H, xe, ye


def _log_norm(H):
    pos = H[H > 0]
    if pos.size == 0:
        return Normalize(vmin=0, vmax=1)
    return LogNorm(vmin=max(1, pos.min()), vmax=H.max())


def _compute_catalog_frangi(catalog, hr_grid, bins=250):
    """Compute Frangi vesselness map from catalog once at startup.
    Returns (fr, xe, ye) or None on failure.
    """
    if catalog is None:
        return None
    try:
        from skimage.filters import frangi as _skimage_frangi
        cmin, cmax, _, mmin, mmax, _ = hr_grid
        H, xe, ye = _hist2d(np.array(catalog["bp_rp"]),
                             np.array(catalog["absG"]),
                             bins, cmin, cmax, mmin, mmax)
        Hs = gaussian_filter(H, sigma=1.5)
        mx = Hs.max()
        if mx == 0:
            return None
        sigmas = list(range(1, 10, 2))
        fr = _skimage_frangi(Hs / mx, sigmas=sigmas, beta=0.3, black_ridges=False)
        return fr, xe, ye
    except Exception:
        return None


def draw_hr_heatmap(fig, df, bins, log_scale, show_contours,
                    show_age_contours, base_model, hr_grid, catalog_frangi=None):
    """Main HR diagram tab — simulation density heatmap.

    Shows the simulated WD population as a 2D color/magnitude density map.
    Optionally overlays WD cooling model mass contours and Frangi ridge
    contours derived from the catalog.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    cmin, cmax, _, mmin, mmax, _ = hr_grid
    w = df["weight"].values if "weight" in df.columns else None

    H, xe, ye = _hist2d(df["bprp"].values, df["G"].values, bins, cmin, cmax, mmin, mmax, w)
    Hm   = np.ma.masked_where(H <= 0, H)
    norm = _log_norm(H) if log_scale else Normalize(vmin=0, vmax=max(H.max(), 1))
    im   = ax.imshow(Hm.T, origin="lower",
                     extent=[xe[0], xe[-1], ye[0], ye[-1]],
                     aspect="auto", cmap="hot", norm=norm, interpolation="nearest")
    _setup_hr_ax(ax, cmin, cmax, mmin, mmax)
    ax.set_title("HR Diagram (Density Map)", fontsize=12, color="white")
    cb = fig.colorbar(im, ax=ax,
                      label="Count (log)" if log_scale else "Count")
    cb.ax.yaxis.set_tick_params(color="white")
    cb.ax.yaxis.label.set_color("white")

    if show_contours and base_model is not None:
        try:
            cs = ax.contour(
                np.nan_to_num(base_model["grid_HR_to_mass"].T),
                levels=[0.4, 0.5, 0.6, 0.9, 1.1, 1.4],
                linestyles="solid", colors="cyan", alpha=0.6, linewidths=0.8,
                extent=(cmin, cmax, mmin, mmax))
            locs = [(1.3, 15.1), (0.6, 14.8), (0.4, 14)]
            ax.clabel(cs, inline=True, inline_spacing=13, rightside_up=True,
                      manual=locs, fontsize=8, fmt=lambda v: f"{v:.1f} M☉")
            ax.text(-0.56, 12, "1.4 M☉", rotation=-63,
                    fontsize=8, color="cyan", alpha=0.6)
        except Exception:
            pass

    if show_age_contours and base_model is not None:
        try:
            age_cs = ax.contour(
                np.nan_to_num(base_model["grid_HR_to_age"].T),
                levels=[1, 2, 3, 4, 6, 9, 13],
                linestyles="dashed", colors="White", alpha=0.9, linewidths=0.8,
                extent=(cmin, cmax, mmin, mmax))
            ax.clabel(age_cs, inline=True, fontsize=8,
                      fmt=lambda v: f"{v:.0f} Gyr")
        except Exception:
            pass

    if catalog_frangi is not None:
        try:
            fr, xe, ye = catalog_frangi
            m_fr = np.where(fr > 0.05, fr, 0)
            nz = m_fr[m_fr > 0]
            if nz.size > 0:
                levels = np.percentile(nz, [30, 60, 95])
                ax.contour(m_fr.T, levels=levels, linestyles="solid",
                           colors="pink", alpha=0.7, linewidths=1.5,
                           extent=(xe[0], xe[-1], ye[0], ye[-1]),
                           origin="lower")
        except Exception:
            pass

    fig.tight_layout()
    return ax


def _shared_norm(H_a, H_b, log_scale):
    """Build a single Normalize/LogNorm spanning both histograms."""
    both = np.concatenate([H_a.ravel(), H_b.ravel()]) if H_b is not None else H_a.ravel()
    vmax = max(both.max(), 1)
    if log_scale:
        pos = both[both > 0]
        vmin = max(1, pos.min()) if pos.size > 0 else 1
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=0, vmax=vmax)


def draw_density_comparison(fig, sim_df, catalog, bins, log_scale, hr_grid):
    """Side-by-side simulation vs. catalog density comparison.

    Left panel: simulated population density (normalized to catalog total).
    Right panel: observed Gaia catalog density. Shared color scale.

    Simplified from the GUI version: no in-place figure caching / selection
    overlays (the web app always draws onto a fresh figure).
    """
    cmin, cmax, _, mmin, mmax, _ = hr_grid
    fig.clear()
    ax_sim, ax_cat = fig.subplots(1, 2, sharex=True, sharey=True)

    H_cat = None
    if catalog is not None:
        H_cat, xe_c, ye_c = _hist2d(np.array(catalog["bp_rp"]),
                                     np.array(catalog["absG"]),
                                     bins, cmin, cmax, mmin, mmax)

    w = sim_df["weight"].values if "weight" in sim_df.columns else None
    H_sim, xe, ye = _hist2d(sim_df["bprp"].values, sim_df["G"].values,
                             bins, cmin, cmax, mmin, mmax, w)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]

    if H_cat is not None:
        sim_sum = H_sim.sum()
        cat_sum = H_cat.sum()
        H_sim_norm = H_sim * (cat_sum / sim_sum) if sim_sum > 0 else H_sim
    else:
        H_sim_norm = H_sim

    shared = _shared_norm(H_sim_norm, H_cat, log_scale)

    Hs_m = np.ma.masked_where(H_sim_norm <= 0, H_sim_norm)
    im_s = ax_sim.imshow(Hs_m.T, origin="lower", extent=extent,
                         aspect="auto", cmap="hot", norm=shared, interpolation="nearest")
    _setup_hr_ax(ax_sim, cmin, cmax, mmin, mmax)
    ax_sim.set_title("Simulation Density", fontsize=11, color="white")
    fig.colorbar(im_s, ax=ax_sim, label="Count")

    if H_cat is not None:
        Hc_m = np.ma.masked_where(H_cat <= 0, H_cat)
        extent_c = [xe_c[0], xe_c[-1], ye_c[0], ye_c[-1]]
        im_c = ax_cat.imshow(Hc_m.T, origin="lower", extent=extent_c,
                             aspect="auto", cmap="hot", norm=shared, interpolation="nearest")
        _setup_hr_ax(ax_cat, cmin, cmax, mmin, mmax)
        ax_cat.set_title("Catalog Density", fontsize=11, color="white")
        fig.colorbar(im_c, ax=ax_cat, label="Count")

    fig.suptitle("Simulation vs. Catalog Density", fontsize=12, color="white")
    fig.tight_layout()
    return ax_sim, ax_cat


def draw_residual_map(fig, sim_df, catalog, bins, hr_grid):
    """Relative residual map between simulation and catalog.

    Computes (sim − cat) / cat per HR bin after normalizing both to probability
    distributions. Symmetric log color scale (blue = sim under-dense,
    red = sim over-dense relative to catalog).
    """
    fig.clear()
    ax = fig.add_subplot(111)
    cmin, cmax, _, mmin, mmax, _ = hr_grid

    if catalog is None:
        ax.text(0.5, 0.5, "No catalog loaded", transform=ax.transAxes,
                ha="center", va="center", color="white")
        return ax

    w = sim_df["weight"].values if "weight" in sim_df.columns else None
    H_sim, xe, ye = _hist2d(sim_df["bprp"].values, sim_df["G"].values,
                             bins, cmin, cmax, mmin, mmax, w)
    H_cat, _, _  = _hist2d(np.array(catalog["bp_rp"]),
                            np.array(catalog["absG"]),
                            bins, cmin, cmax, mmin, mmax)

    Hs_p = H_sim / max(H_sim.sum(), 1e-10)
    Hc_p = H_cat / max(H_cat.sum(), 1e-10)
    eps  = Hc_p.max() * 1e-3
    diff = (Hs_p - Hc_p) / (Hc_p + eps)

    cmap_rb = LinearSegmentedColormap.from_list(
        "rb", ["#003CFF", "#000000", "#F30000"])
    mv   = np.abs(diff).max()
    norm = SymLogNorm(linthresh=max(mv / 100, 1e-6), vmin=-mv, vmax=mv, base=10)
    im   = ax.imshow(diff.T, origin="lower",
                     extent=[xe[0], xe[-1], ye[0], ye[-1]],
                     aspect="auto", cmap=cmap_rb, norm=norm, interpolation="nearest")
    _setup_hr_ax(ax, cmin, cmax, mmin, mmax)
    ax.set_title("Residual Map (Relative)", fontsize=12, color="white")
    cb = fig.colorbar(im, ax=ax, label="Relative Residual: (sim−cat)/cat")
    cb.ax.yaxis.set_tick_params(color="white")
    cb.ax.yaxis.label.set_color("white")
    fig.tight_layout()
    return ax


def draw_distributions(fig, df):
    """WD mass, age, and merger-mass histograms (three side-by-side panels)."""
    fig.clear()
    ax1, ax2, ax3 = fig.subplots(1, 3)
    w        = df["weight"].values if "weight" in df.columns else None
    weighted = w is not None and not np.allclose(w, 1.0)
    ylabel   = "Weighted Count" if weighted else "Count"

    ax1.hist(df["Mass"], bins=50, color="steelblue",
             alpha=0.7, edgecolor="black", weights=w)
    ax1.set_xlabel("Mass [M☉]", fontsize=10)
    ax1.set_ylabel(ylabel, fontsize=10)
    ax1.set_title("Mass Distribution (all)", fontsize=11)
    ax1.grid(alpha=0.3)

    ax2.hist(df["true_age"], bins=50, color="coral",
             alpha=0.7, edgecolor="black", weights=w)
    ax2.set_xlabel("Age [Gyr]", fontsize=10)
    ax2.set_ylabel(ylabel, fontsize=10)
    ax2.set_title("Age Distribution", fontsize=11)
    ax2.grid(alpha=0.3)

    if "merger_flag" in df.columns:
        merger_mask = df["merger_flag"].values == 1
        mm = df["Mass"].values[merger_mask]
        mw = w[merger_mask] if w is not None else None
        if mm.size > 0:
            ax3.hist(mm, bins=50, color="indianred",
                     alpha=0.7, edgecolor="black", weights=mw)
        else:
            ax3.text(0.5, 0.5, "No mergers in selection",
                     ha="center", va="center", transform=ax3.transAxes,
                     color="#888", fontsize=10)
    else:
        ax3.text(0.5, 0.5, "merger_flag missing",
                 ha="center", va="center", transform=ax3.transAxes,
                 color="#888", fontsize=10)
    ax3.set_xlabel("Mass [M☉]", fontsize=10)
    ax3.set_ylabel(ylabel, fontsize=10)
    ax3.set_title("Mass Distribution (mergers)", fontsize=11)
    ax3.grid(alpha=0.3)

    fig.tight_layout()
