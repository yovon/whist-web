"""simulate_wd_full_population.py

Simulate synthetic white dwarf (WD) populations and derive observables via
cooling-track model grids.

Summary
-------
This module provides:
- create_population(params): Monte Carlo creation of a WD population. Produces a
  pandas DataFrame with mass, cooling/true ages and per-object flags:
    - type_B (atmosphere family), atm_thickness_flag (thick/thin),
      core_flag/core_type (CO/Ne), distil_flag, merger_flag, ...
  Units: Mass [Msun], ages [Gyr].

- extract_observables(all_data, params): Given the population DataFrame,
  compute:
    - teff (K) using model interpolation (mass, cooling_age -> log10(Teff))
    - photometry (absolute Mag and color) using (mass, log10(Teff)) grids
    - distances (pc), apparent magnitudes and parallax; apply Gaia-like cuts.
  Uses WDModelsLoader to pick model tuples and cache interpolators; groups
  stars by chosen tuple to evaluate interpolators in bulk (vectorized).

- simulate_HR_all(params): High-level wrapper that runs creation + extraction,
  plots an HR diagram and saves the selected objects.

Design & important implementation notes
- Photometry grids expect log10(Teff) on the y-axis. Always pass np.log10(teff)
  when calling photometry interpolators.
- Use loader.get_interpolator(tpl, output_key, y_axis=...) to obtain the
  (mass, y) -> output interpolator. This respects caching and correct y-axis
  selection.
- Distilled (special-cooling) sequences are handled separately via
  loader.get_distil_teff(...).
- The implementation groups by model tuple to avoid per-star Python-level loops
  for interpolation, improving performance.

See README.md for detailed physics background, algorithmic explanation and
usage examples.
"""

import sys, os
import numpy as np
import matplotlib.pylab as plt
from astropy import constants as cs
import scipy.integrate as sci
from scipy.interpolate import interp1d
from scipy.interpolate import griddata
import pandas as pd
import WD_models_loader_wrapper
from c_enrichment_prescriptions import CEnrichmentModel as _CEnrichmentModel
from selection_models import SelectionModel as _SelectionModel, AbsoluteMagCap as _AbsoluteMagCap
from error_models import ErrorModel as _ErrorModel
import json
import time
import re

import warnings
warnings.filterwarnings('ignore')
np.seterr(divide='ignore', invalid='ignore', over='ignore')


def plot_HR(bprp,G, age, alpha=1, model=None, color='gray', grid=True):
      
    ax = plt.subplot(1,1,1); plt.clf()
    plt.axes()
    plt.scatter(bprp, G, c=age, vmin=0,vmax=7,cmap='cool', marker='o', s=1, alpha=alpha)
    plt.gca().invert_yaxis()
    age_colorbar_param={'pad':-1, 'aspect':10,'shrink':0.5, 'fraction':1,'anchor':(10,1),'panchor':False,
                        'extend':'max','ticks':(0,2,4,6)}
    plt.colorbar(orientation='horizontal',**age_colorbar_param)
    # plt.sca(ax)
    plt.xlabel('$\\rm{G_{BP} - G_{RP}}$')
    plt.ylabel('$\\rm{M_G}$')
    
    #grid
    if grid == True:
        HR_grid    = (-0.5, 2, 0.002, 9, 16, 0.01)
        plt.contour(np.nan_to_num(model['grid_HR_to_mass'].T),levels=[0.5,0.9,1.1,1.4], linestyles='solid',cmap='jet', alpha=alpha, vmax=1000, zorder=1, extent=(HR_grid[0],HR_grid[1],HR_grid[3],HR_grid[4]))
                       

        CS = plt.contour(model['grid_HR_to_age'].T,
                         levels=[1,2,3,4,6,9,13],
                         linestyles='solid', cmap='jet',
                         alpha=alpha,vmin=-1000,zorder=3,
                       extent=(HR_grid[0],HR_grid[1],HR_grid[3],HR_grid[4]))
        
        # label of mass contour
        plt.text(-0.39,10.1,'0.5 M☉',rotation=-50,fontsize=15,color=(0,0,0.5,1.0),alpha=alpha)
        # plt.text(-0.46,10.3,'0.7 M☉',rotation=-50,fontsize=15,color=(0,0,0.5,1.0),alpha=alpha)
        plt.text(-0.51,10.5,'0.9 M☉',rotation=-55,fontsize=15,color=(0,0,0.5,1.0),alpha=alpha)
        plt.text(-0.57,10.7,'1.1 M☉',rotation=-60,fontsize=15,color=(0,0,0.5,1.0),alpha=alpha)
        plt.text(-0.56,12,'1.4 M☉',rotation=-60,fontsize=15,color=(0,0,0.5,1.0),alpha=alpha)
        # label of age contour
        plt.text(-0.32+0.05,13.25,'$1\\rm{Gyr}$',rotation=70,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
        plt.text(-0.045+0.05,13.9,'$2\\rm{Gyr}$',rotation=80,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
        plt.text(0.21+0.05,14.8,'$3\\rm{Gyr}$',rotation=90,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
        plt.text(0.42+0.05,14.7,'$4\\rm{Gyr}$',rotation=115,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
        plt.text(0.58+0.05,14.7,'$5\\rm{Gyr}$',rotation=60,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
        plt.text(0.77+0.05,14.83,'$6\\rm{Gyr}$',rotation=40,fontsize=15,color=(0.5,0,0.0,1.0),alpha=alpha)
    
    return ax


def save_data(arrays, file_name, names=None):
    print("Saving Data")
    BASE_DIR = os.getcwd()
    full_path = os.path.join(BASE_DIR, "Simulations", file_name)

    if type(arrays).__name__ == "DataFrame":
        arrays.to_csv(full_path)
        return
    
    # If not dataframe, assume a list of numpy arrays
    df = pd.DataFrame(np.matrix(arrays).T)
    if names is None:
        names = [str(i) for i in range(len(arrays))]
    df.columns = names
    df.to_csv(full_path)


def save_df_with_metadata(df, file_name, metadata=None, base_dir=None):
    """
    Save DataFrame to CSV and prepend metadata as commented JSON lines.
    - metadata: dict (will be json-dumped). Keys with non-serializable values are cast to str.
    The CSV remains a valid CSV; readers that ignore leading '#' comment lines will skip the metadata.
    """
    if base_dir is None:
        base_dir = os.getcwd()
    out_dir = os.path.join(base_dir, "Simulations")
    os.makedirs(out_dir, exist_ok=True)
    full_path = os.path.join(out_dir, file_name)

    meta = metadata or {}
    try:
        meta_json = json.dumps(meta, indent=2, default=str)
    except Exception:
        # fallback: convert values to strings
        meta_json = json.dumps({k: str(v) for k, v in meta.items()}, indent=2)

    import platform

    if platform.system() == 'Windows':
        newline = '\r\n'
    else:
        newline = '\n'

    # write metadata as commented lines, then CSV
    with open(full_path, "w", encoding="utf-8", newline=newline) as f:
        for line in meta_json.splitlines():
            f.write("# " + line + newline)
        f.write("# --- DATA ---" + newline)
        df.to_csv(f, index=False)

    return full_path


def sample_merger_times(n, t_min, t_max, base=np.e, rng=None):
    rng = np.random.default_rng(rng)
    log_min = np.log(t_min) / np.log(base)
    log_max = np.log(t_max) / np.log(base)
    u = rng.uniform(log_min, log_max, size=n)
    return base ** u  # times with p(t) ∝ 1/t on [t_min, t_max]


def stars_per_burst(p, rng=None):
    """
    Factory: return a callable f(t) -> integer number of stars born at time t (years).
    - p: params dict (supports 'sfr_type','burst_N','sfr_tau','sfr_func', etc.)
    - rng: optional numpy Generator for reproducible Poisson draws; if None returns a deterministic callable.

    The returned function samples the number of stars per burst using a Poisson draw
    around the expected baseline * r(t). If rng is None the callable will return
    int(round(expected)) deterministically.

    Supported sfr_type: 'constant', 'exponential', or a custom callable via 'sfr_func'.
    """
    if rng is None:
        rng_local = None
    else:
        rng_local = rng

    # build relative SFR function r(t)
    if "sfr_func" in p and callable(p["sfr_func"]):
        def r_of_t(t):
            return float(p["sfr_func"](t, p))
    else:
        typ = p.get("sfr_type", "constant")
        if typ == "constant":
            def r_of_t(t):
                return 1.0
        elif typ == "exponential":
            tau = float(p.get("sfr_tau", 5e9))
            def r_of_t(t):
                return np.exp(-t / tau)
        elif typ == "1_over_t":
            t_ref = float(p.get("sfr_tau", 5e9))
            def r_of_t(t):
                return t_ref / max(t, 1.0)
        else:
            # unknown type -> constant fallback
            def r_of_t(t):
                return 1.0

    baseline = max(0.0, float(p.get("burst_N", 50)))

    if rng_local is None:
        # deterministic callable
        def f(t):
            expected = baseline * r_of_t(t)
            if expected < 0.5:
                return 0
            return int(np.round(expected))
    else:
        # stochastic callable using Poisson sampling for realism
        def f(t):
            expected = baseline * r_of_t(t)
            if expected <= 0.0:
                return 0
            return int(rng_local.poisson(expected))

    return f


def hurley_tms(M, Z=0.02):
    """Hurley, Pols & Tout 2000 (MNRAS 315, 543, eqs 4-7) MS lifetime in Gyr,
    with metallicity dependence via zeta = log10(Z / 0.02).

    The a1..a10 coefficients are the zeta-polynomials of Appendix A (SSE `zcnsts`).
    At Z = 0.02 (zeta = 0) this reduces exactly to the solar coefficients in
    tests/investigations/_ab_common.hurley (used as a regression check).
    Vectorised: M (and optionally Z) may be scalars or numpy arrays.
    """
    M = np.asarray(M, dtype=float)
    Z = np.asarray(Z, dtype=float)
    zeta = np.log10(Z / 0.02)
    z2 = zeta * zeta
    z3 = z2 * zeta
    a1 = 1.593890e3 + 2.053038e3 * zeta + 1.231226e3 * z2 + 2.327785e2 * z3
    a2 = 2.706708e3 + 1.483131e3 * zeta + 5.772723e2 * z2 + 7.411230e1 * z3
    a3 = 1.466143e2 - 1.048442e2 * zeta - 6.795374e1 * z2 - 1.391127e1 * z3
    a4 = 4.141960e-2 + 4.564888e-2 * zeta + 2.958542e-2 * z2 + 5.571483e-3 * z3
    a5 = 3.426349e-1
    a6 = 1.949814e1 + 1.758178e0 * zeta - 6.008212e0 * z2 - 4.470533e0 * z3
    a7 = 4.903830e0
    a8 = 5.212154e-2 + 3.166411e-2 * zeta - 2.750074e-3 * z2 - 2.271549e-3 * z3
    a9 = 1.312179e0 - 3.294936e-1 * zeta + 9.231860e-2 * z2 + 2.610989e-2 * z3
    a10 = 8.073972e-1
    tbgb = (a1 + a2 * M**4 + a3 * M**5.5 + M**7) / (a4 * M**2 + a5 * M**7)
    mu = np.maximum(0.5, 1.0 - 0.01 * np.maximum(a6 / M**a7, a8 + a9 / M**a10))
    thook = mu * tbgb
    x = np.maximum(0.95, np.minimum(0.95 - 0.03 * (zeta + 0.30103), 0.99))
    return 1e6 * np.maximum(thook, x * tbgb) / 1e9


def expected_births(t, p):
    """Expected (deterministic) number of stars born at burst time t [yr].

    = burst_N * r(t) + extra_bump term. Accepts scalar or array t and returns the
    same shape. This mirrors the per-burst count logic in create_population but
    WITHOUT Poisson noise, rounding, or importance-sampling oversampling — it is the
    smooth star-formation history used to build the chemical-enrichment AMR.
    """
    t = np.asarray(t, dtype=float)
    baseline = max(0.0, float(p.get("burst_N", 50)))
    if "sfr_func" in p and callable(p["sfr_func"]):
        flat = np.atleast_1d(t)
        r = np.array([float(p["sfr_func"](float(ti), p)) for ti in flat]).reshape(t.shape)
    else:
        typ = p.get("sfr_type", "constant")
        if typ == "exponential":
            tau = float(p.get("sfr_tau", 5e9))
            r = np.exp(-t / tau)
        elif typ == "1_over_t":
            t_ref = float(p.get("sfr_tau", 5e9))
            r = t_ref / np.maximum(t, 1.0)
        else:
            r = np.ones_like(t)
    n = baseline * r
    eb = p.get("extra_bump")
    if eb:
        n = n + eb["amp_fraction"] * baseline * np.exp(
            -0.5 * ((t - eb["t_center"]) / eb["sigma"]) ** 2)
    return n


def build_metallicity_amr(p):
    """Build the chemical-enrichment age-metallicity relation Z(t).

    Z is proportional to the cumulative number of stars formed up to birth time t
    (a closed-box enrichment proxy), normalised so the most recent burst sits at
    amr_z_present (solar by default). Older bursts come out metal-poor, with the
    shape following the active SFR schema (sfr_type + extra_bump). Returns a
    vectorised callable Z_of_t(t) via linear interpolation over the time grid.
    """
    times = np.arange(int(p["time_start"]), int(p["time_stop"]),
                      int(p["time_step"]), dtype=float)
    z_present = float(p.get("amr_z_present", 0.02))
    z_floor = float(p.get("amr_z_floor", 1e-3))
    if times.size == 0:
        return lambda t: np.full_like(np.asarray(t, dtype=float), z_present)
    cum = np.cumsum(expected_births(times, p))
    total = cum[-1] if cum[-1] > 0 else 1.0
    z_grid = np.maximum(z_floor, z_present * (cum / total))
    return interp1d(times, z_grid, kind="linear", bounds_error=False,
                    fill_value=(z_grid[0], z_grid[-1]))


def create_population(params, show_times=False):
    """Create WD population and return a DataFrame with initial flags/columns.
    Returns: all_data (DataFrame)

    Vectorized: instead of looping over bursts and concatenating, build a
    per-star birth-time array up front (np.repeat) and run the whole population
    through a single vectorized pass. Statistically equivalent to the old
    burst-loop version (archived in archive/create_population_loop.py) but ~6x
    faster; RNG draw order differs, so output is not bit-identical for a seed.
    """
    p = params
    t0 = time.perf_counter()

    # --- IMFs and helper math (same as before) ---
    IMF = lambda m: m**p["IMF_exponent"]  # Salpeter-like IMF: dN/dM ∝ M^(-alpha), default -2.3
    if p.get("IFMR_mode") == "piecewise":
        # IFMR_bp_x/y are the authoritative lists when MCMC is active;
        # fall back to IFMR_breakpoints for GUI-only use
        if p.get("IFMR_bp_x") and p.get("IFMR_bp_y"):
            _xs = list(p["IFMR_bp_x"])
            _ys = list(p["IFMR_bp_y"])
        else:
            _xs = [b[0] for b in p["IFMR_breakpoints"]]
            _ys = [b[1] for b in p["IFMR_breakpoints"]]
        M_final = interp1d(_xs, _ys, kind="linear", fill_value="extrapolate", bounds_error=False)
    else:
        M_final = lambda m: p["IFMR_parameters"][0] + p["IFMR_parameters"][1] * m
    T_ms = lambda m: 10*(m)**-2.5  # Gyr -> MS lifetime (solar power law)

    # MS-lifetime law and metallicity AMR. Enabling the AMR forces the Hurley law,
    # since the power law has no metallicity dependence (AMR would be a no-op).
    ms_lifetime_law = p.get("ms_lifetime_law", "powerlaw")
    use_amr = bool(p.get("use_amr", False))
    amr_scatter = float(p.get("amr_scatter", 0.0))
    Z_of_t = None
    if use_amr:
        ms_lifetime_law = "hurley"
        Z_of_t = build_metallicity_amr(p)

    mass_range = p["ms_mass_range"]  # MS mass range set directly
    A = 1 / sci.quad(IMF, mass_range[0], mass_range[1])[0]
    # Inverse-CDF sampling of dN/dM ∝ m^alpha on [m_lo, m_hi]. With e = alpha+1 the
    # CDF is F(m) = A*(m^e - m_lo^e)/e, so m(u) = (m_lo^e + u*e/A)^(1/e). A above is
    # already the correct PDF normalisation for ANY alpha. Previously the e=-1.3 of
    # the default alpha=-2.3 was hardcoded here, so varying IMF_exponent only
    # rescaled the upper-mass cutoff via A — it did not re-slope the progenitor IMF.
    # e≈0 (alpha≈-1) is the log-uniform limit, handled separately to avoid 1/e.
    _imf_e = p["IMF_exponent"] + 1.0
    if abs(_imf_e) < 1e-9:
        IMF_inverse = lambda u: mass_range[0] * (mass_range[1] / mass_range[0]) ** u
    else:
        IMF_inverse = lambda u: (mass_range[0] ** _imf_e + u * _imf_e / A) ** (1.0 / _imf_e)

    rng = np.random.default_rng(p.get("seed", None))

    if p.get("ne_core_fraction", 0) > 0:
        import warnings
        warnings.warn(
            "ne_core_fraction is no longer used — core assignment is now deterministic "
            "based on ne_core_mass_threshold and ne_core_mass_threshold_cohe.",
            DeprecationWarning, stacklevel=2,
        )

    # Channel fractions + secondary-mass sampler (validated/defined once; were
    # previously re-checked/re-defined every burst).
    dwd_frac  = float(p.get("dwd_fraction", 0.292))
    wdsg_frac = float(p.get("wdsg_fraction", 0.008))
    if dwd_frac + wdsg_frac > 1.0:
        raise ValueError(
            f"dwd_fraction + wdsg_fraction must be <= 1 (got {dwd_frac + wdsg_frac:.3f})"
        )
    # Secondary mass sampling — per-channel power law p(M_2) ∝ M_2^-alpha
    # on [m_min, M_primary]. alpha=0 reproduces uniform [m_min, M_primary].
    # A small absolute floor avoids the divergence at M=0 for alpha >= 1.
    dwd_alpha  = float(p.get("dwd_imf_alpha",  0.0))
    wdsg_alpha = float(p.get("wdsg_imf_alpha", 0.0))
    m_min_secondary = 1e-3  # M☉ — numerical floor for the power-law sample

    def _powerlaw_sample(M_primary_arr, alpha, rng):
        n = M_primary_arr.size
        u = rng.random(n)
        lo = m_min_secondary
        hi = M_primary_arr
        if abs(alpha - 1.0) < 1e-9:
            return lo * (hi / lo) ** u
        e = 1.0 - alpha
        return ((hi**e - lo**e) * u + lo**e) ** (1.0 / e)

    # --- Step 0: per-burst integer star counts. Reuse stars_per_burst so the SFR
    # schema stays in one place; the extra_bump term is added here exactly as the
    # loop did. This is a cheap scalar pass over BURSTS only — the per-star physics
    # below stays fully vectorized.
    ts = np.arange(int(p["time_start"]), int(p["time_stop"]), int(p["time_step"]), dtype=float)
    sfr = stars_per_burst(p)
    # np.intp (platform int: int64 native, int32 under 32-bit WASM/Pyodide) so the
    # counts can be passed straight to np.repeat as `repeats` without an unsafe
    # int64->int32 cast that fails in the browser.
    N_born_burst = np.fromiter((sfr(t) for t in ts), dtype=np.intp, count=ts.size)
    if "extra_bump" in p:
        eb = p["extra_bump"]
        bump = (eb["amp_fraction"] * p.get("burst_N", 200)
                * np.exp(-0.5 * ((ts - eb["t_center"]) / eb["sigma"]) ** 2))
        N_born_burst = N_born_burst + np.maximum(0, np.round(bump).astype(np.intp))

    # importance sampling: oversample at late bursts (small time_left) to improve
    # statistics for old WDs. weight = time_left / time_stop so weighted counts
    # recover true numbers.
    if p.get("use_importance_sampling", False):
        time_left_eff = np.maximum(p["time_stop"] - ts, int(p["time_step"]))  # avoid /0 near end
        oversample = p["time_stop"] / time_left_eff
        N_born_burst = (N_born_burst * oversample).astype(np.intp)  # truncates, like int(...)
        burst_weight = time_left_eff / p["time_stop"]
    else:
        burst_weight = np.ones_like(ts)

    # --- Step 1: expand burst-level scalars to per-star arrays.
    t_star      = np.repeat(ts, N_born_burst)
    star_weight = np.repeat(burst_weight, N_born_burst)
    time_left   = p["time_stop"] - t_star
    N = t_star.size

    # --- Step 2: per-star physics (identical math, now over the whole population).
    u = rng.random(N)
    ms_mass = IMF_inverse(u)

    # Per-star metallicity: AMR-driven (mean Z(t) is constant within a burst, with
    # optional per-star dex scatter) or solar when AMR is off.
    if use_amr:
        Z_star = np.asarray(Z_of_t(t_star), dtype=float)
        if amr_scatter > 0:
            Z_star = 0.02 * 10 ** (np.log10(Z_star / 0.02)
                                   + rng.normal(0.0, amr_scatter, N))
        Z_star = np.clip(Z_star, 1e-4, 0.03)  # Hurley validity range
    else:
        Z_star = np.full(N, 0.02)

    if ms_lifetime_law == "hurley":
        MS_lifetime = hurley_tms(ms_mass, Z_star) * 1e9  # years
    else:
        MS_lifetime = T_ms(ms_mass) * 1e9  # years

    wd_mass = M_final(ms_mass)
    cool_age = time_left - MS_lifetime  # the time the wd have to cool down
    true_age = np.array(time_left, dtype=float)
    distil = np.zeros(N, dtype=bool)

    # Step A: assign DWD and WD+SG channels (independent fractions, both over the
    # whole population).
    u_channel = rng.random(N)
    is_dwd   = u_channel < dwd_frac
    is_wdsg  = (u_channel >= dwd_frac) & (u_channel < dwd_frac + wdsg_frac)

    companion_mass = np.zeros(N)
    if is_dwd.any():
        companion_mass[is_dwd]  = _powerlaw_sample(wd_mass[is_dwd],  dwd_alpha,  rng)
    if is_wdsg.any():
        companion_mass[is_wdsg] = _powerlaw_sample(wd_mass[is_wdsg], wdsg_alpha, rng)

    binaries = is_dwd | is_wdsg

    # Step B: companion atmosphere type (DWD only)
    companion_atm = (rng.random(N) < p["type_b_fraction"]).astype(int)
    companion_atm[~is_dwd] = 0  # only meaningful for DWD

    # Step C: delay times — DWD uses 1/t, WD+SG uses a constant delay
    merger_time = np.full(N, np.inf)
    n_dwd  = int(np.count_nonzero(is_dwd))
    n_wdsg = int(np.count_nonzero(is_wdsg))
    if n_dwd > 0:
        merger_time[is_dwd]  = sample_merger_times(n_dwd, 1e3, 1e10, base=10, rng=rng)
    if n_wdsg > 0:
        merger_time[is_wdsg] = float(p.get("wdsg_delay", 1e7))

    # Step D: merged masks
    merged_dwd  = is_dwd  & (MS_lifetime + merger_time < time_left)
    merged_wdsg = is_wdsg & (MS_lifetime + merger_time < time_left)
    merged = merged_dwd | merged_wdsg

    # Step E: kinematics + merger remnant mass — direct sum of primary + secondary.
    cool_age[merged] -= merger_time[merged]
    wd_mass[merged]  = wd_mass[merged] + companion_mass[merged]
    companion_mass[merged] = 0.0  # merged companions absorbed into primary

    # Step F: atmosphere draws, then deterministic core composition
    atm_type   = (rng.random(N) < p["type_b_fraction"]).astype(int)
    atm_thin = (rng.random(N) < p["atm_thin_fraction"]).astype(int)

    thr     = p["ne_core_mass_threshold"]
    thr_coh = p.get("ne_core_mass_threshold_cohe", 1.20)
    # CO+CO: both primary and companion are DA; CO+He: at least one non-DA
    is_coco = merged_dwd & (atm_type == 0) & (companion_atm == 0)
    is_cohe = merged_dwd & ~is_coco

    core_flag = np.zeros(N, dtype=bool)  # default is CO (0), Ne is 1
    core_flag[~binaries]   = wd_mass[~binaries]   >= thr      # singles
    core_flag[is_coco]     = wd_mass[is_coco]     >= thr      # DWD CO+CO
    core_flag[is_cohe]     = wd_mass[is_cohe]     >= thr_coh  # DWD CO+He
    core_flag[merged_wdsg] = wd_mass[merged_wdsg] >= thr      # WD+SG

    # Step G: distillation — only WD+SG mergers are eligible
    # NOTE: distil over 1.26 M☉ will be null later (no STELUM tracks above 1.26 M☉)
    # NOTE: the low limit at 1.0 M☉ is because this is the lowest STELLUM track available
    eligible_distil = merged_wdsg & (wd_mass > 1.0)
    n_elig = int(np.count_nonzero(eligible_distil))
    if n_elig > 0:
        distil[eligible_distil] = rng.random(n_elig) < p["distil_probability"]

    # Step H: merger_channel encoding
    channel = np.zeros(N, dtype=int)  # 0 = single
    channel[merged_dwd]        = 1          # merged DWD
    channel[merged_wdsg]       = 2          # merged WD+SG
    channel[is_dwd  & ~merged] = 3          # non-merged DWD binary
    channel[is_wdsg & ~merged] = 4          # non-merged WD+SG (rare)

    have_age = (cool_age >= 0)
    over_mass_mergers = (wd_mass > 1.4) & merged # TODO: can use this in the future to flag potential mergers that become NS
    mask = have_age & (~over_mass_mergers)

    # --- Step 3: select valid stars directly (no per-burst concatenation).
    masses = wd_mass[mask]
    cool_ages = cool_age[mask] * 1e-9
    true_ages = true_age[mask] * 1e-9
    distil_flag = distil[mask]
    merger_flag = merged[mask].astype(int)
    merger_channel = channel[mask].astype(int)
    companion_atm_flag = companion_atm[mask].astype(int)
    atm_type_flag = atm_type[mask].astype(int)
    atm_thickness_flag = atm_thin[mask].astype(int)
    core_flag = core_flag[mask].astype(int)
    companion_masses = companion_mass[mask]
    weights = star_weight[mask]
    metallicities = Z_star[mask]

    rounded_masses = np.round(masses / 0.05) * 0.05
    teffs = np.full(len(masses), np.nan)

    all_data = pd.DataFrame({
        "Mass": masses,
        "rounded_Mass": rounded_masses,
        "cool_age": cool_ages,
        "true_age": true_ages,
        "distil_flag": distil_flag.astype(int),
        "merger_flag": merger_flag.astype(int),
        "merger_channel": merger_channel,        # 0=single, 1=DWD merged, 2=WD+SG merged, 3=DWD non-merged, 4=WD+SG non-merged
        "companion_atm_flag": companion_atm_flag, # 0=DA, 1=non-DA companion (DWD only)
        "atm_type_flag": atm_type_flag, # 0 = H, 1 = He
        "atm_thickness_flag": atm_thickness_flag,   # 0 = thick (be), 1 = thin (bet)
        "core_flag": core_flag, # 0 = CO, 1 = Ne
        "teff": teffs,
        "companion_mass": companion_masses,  # >0 for non-merged binaries; 0 for singles/merged
        "weight": weights,                   # importance sampling weight (1.0 if disabled)
        "metallicity": metallicities,        # progenitor Z (AMR-driven; 0.02 solar if AMR off)
    })

    # human-readable columns
    all_data["atm_type"] = np.where(all_data["atm_type_flag"] == 0, "H", "He")
    all_data["atm_thickness"] = np.where(all_data["atm_thickness_flag"] == 0, "thick", "thin")
    all_data["core_type"] = np.where(all_data["core_flag"] == 1, "Ne", "CO")
    all_data["c_he_log"] = np.nan   # filled in extract_observables for He-rich WDs

    total_time = time.perf_counter() - t0
    if show_times:
        print(f"Population created with {len(all_data)} WDs in {total_time:.4f} seconds.")

    return all_data


def extract_observables(all_data, params, show_progress=False, progress_callback=None, loader=None, show_timings=False, use_direct_approach=True):
    """
    Compute teff, Mag, color, distances and Gaia-like selection.
    Flow:
      1) Build masks for non-distilled sub-populations (Ne non-merger, He-mid, rest)
      2) For each mask: choose model tuple (via WDModelsLoader), group by tuple and evaluate
      3) Handle distillation targets (teff via distillation grid)
      4) Compute photometry for remaining valid sources, distances and Gaia cuts

    Parameters
    -----------
    all_data : DataFrame
        Population data with masses, ages, flags.
    params : dict
        Simulation/config parameters.
    show_progress : bool
        Retained for API compatibility; progress is reported via progress_callback.
    progress_callback : callable, optional
        Callback function for progress updates.
    loader : WDModelsLoader, optional
        Pre-initialized model loader. If None, creates a new loader.
        Reusing a loader across multiple simulations avoids reloading models.
        Call loader.build_sdq_cache(grid_root) before the first call to avoid
        rebuilding the StealthDQ photometry grid on each invocation.
    show_timings : bool, optional
        If True, prints detailed timing breakdown. Default False.
    use_direct_approach : bool, optional
        If True, tries direct (mass, age)->Mag/color first, then falls back to two-step.
        If False, uses only two-step (mass,age)->teff->Mag/color. Default True.
    """
    # Initialize timing dictionary (optional)
    if show_timings:
        timings = {
            '[0.0] loader_init': 0.0,
            '[0.1] mask_build': 0.0,
            '[1] choose_tuples': 0.0,
            '[2] model_cache_building': 0.0,
            '[3.0] interp_teff': 0.0,
            '[3] photometry': 0.0,
            '[4] distilled': 0.0,
            '[5] c_enrichment': 0.0,
            '[6] binary_companion': 0.0,
            '[7] validity_selection': 0.0,
            'total': 0.0
        }
        t_total_start = time.perf_counter()

    p = params
    mid_low, mid_high = p.get("mid_mass_range", (0.5, 1.0))


    # --- PHASE 0: Initializers and masks ---------------------------------
    # ---------------------------------------------------------------------

    # loader: use provided one or create new (lazy by default so we don't pay cost until needed)
    if show_timings:
        t0 = time.perf_counter()
    if loader is None:
        loader = WD_models_loader_wrapper.WDModelsLoader(HR_grid=p["HR_grid"], HR_bands=p["HR_bands"], lazy=True, interp_method="grid", grid_res=(300,800))
    if show_timings:
        timings['[0.0] loader_init'] = time.perf_counter() - t0
    # Optional pre-load: if you prefer loader.build_cache() uncomment next line
    # loader.build_cache()

    # prepare result columns
    all_data["teff"] = np.nan
    all_data["G"] = np.nan
    all_data["bprp"] = np.nan

    # non-distilled master mask
    if show_timings:
        t0 = time.perf_counter()
    mask_nd = (all_data["distil_flag"] == 0)

    # define sub-masks
    # Exclude DWD merged ONe (channel 1) — no Ne-22/Mg-26, wrong tracks.
    # Singles (0), WD+SG merged (2), and non-merged binaries (3/4) all use ONe tracks normally.
    # Fall back to merger_flag==0 for DataFrames produced before the merger_channel column existed.
    if "merger_channel" in all_data.columns:
        mask_ne = mask_nd & (all_data["core_type"].str.lower().str.contains("ne", na=False)) & (all_data["merger_channel"] != 1)
    else:
        mask_ne = mask_nd & (all_data["core_type"].str.lower().str.contains("ne", na=False)) & (all_data["merger_flag"] == 0)
    mask_he_mid = mask_nd & (all_data["atm_type"].str.lower() == "he") & (all_data["Mass"] >= mid_low) & (all_data["Mass"] < mid_high)
    mask_rest = mask_nd & ~(mask_ne | mask_he_mid)

    if use_direct_approach:
        # He stars are handled entirely in the C-enrichment block below
        _he_excl = all_data["atm_type_flag"] == 1
        mask_ne      = mask_ne      & ~_he_excl
        mask_he_mid  = mask_he_mid  & ~_he_excl  # becomes empty; kept for symmetry
        mask_rest    = mask_rest    & ~_he_excl

    masks = [("ne_nonmerger", mask_ne), ("he_mid", mask_he_mid), ("rest", mask_rest)]
    if show_timings:
        timings['[0.1] mask_build'] = time.perf_counter() - t0

    from collections import defaultdict
    # optional progress: announce start
    if progress_callback:
        try:
            progress_callback(0.0, "Preparing subsets")
        except Exception:
            pass

    weights_nd = 0.8
    per_mask_weight = weights_nd / max(1, len(masks))

    
    # --- PHASE 1: Collect all unique tuples and prepare groups for all masks -------------------------
    # -------------------------------------------------------------------------------------------------

    all_unique_tuples = set()
    mask_groups = []  # Store (mask_index, name, groups) for later processing

    for mask_index, (name, mask) in enumerate(masks):
        idxs = np.where(mask)[0]
        if idxs.size == 0:
            continue

        if show_timings:
            t_choose = time.perf_counter()

        atm_types   = all_data["atm_type_flag"].iloc[idxs].values
        thicknesses = all_data["atm_thickness_flag"].iloc[idxs].values
        core_types  = all_data["core_flag"].iloc[idxs].values

        combinations = np.column_stack([atm_types, thicknesses, core_types])
        unique_rows, inverse = np.unique(combinations, axis=0, return_inverse=True)

        groups = defaultdict(list)
        for i, unique_row in enumerate(unique_rows):
            atm, thick, core = unique_row
            atm   = "He" if atm   == 1 else "H"
            thick = "thin" if thick == 1 else "thick"
            core  = "Ne"  if core  == 1 else "CO"
            tpl = loader.choose_tuple(atm=atm, thickness=thick, core=core, mass_is_midrange=True)
            all_unique_tuples.add(tpl)
            group_idxs = idxs[inverse == i]
            if len(group_idxs) > 0:
                groups[tpl].extend(group_idxs.tolist())

        if show_timings:
            timings['[1] choose_tuples'] += time.perf_counter() - t_choose

        mask_groups.append((mask_index, name, groups))

    # Also collect unique tuples for He non-distilled stars (handled in c-enrichment, not Phase 3)
    if use_direct_approach:
        if show_timings:
            t_choose = time.perf_counter()
        _he_nd_idxs_p1 = np.where((all_data["atm_type_flag"] == 1) & mask_nd)[0]
        if _he_nd_idxs_p1.size > 0:
            _he_masses_p1 = all_data["Mass"].iloc[_he_nd_idxs_p1].values
            _he_combos = np.column_stack([
                all_data["atm_type_flag"].iloc[_he_nd_idxs_p1].values,
                all_data["atm_thickness_flag"].iloc[_he_nd_idxs_p1].values,
                all_data["core_flag"].iloc[_he_nd_idxs_p1].values,
                ((_he_masses_p1 >= mid_low) & (_he_masses_p1 < mid_high)).astype(int),
            ])
            for _row in np.unique(_he_combos, axis=0):
                _atm, _thick, _core, _mid = _row
                all_unique_tuples.add(loader.choose_tuple(
                    "He" if _atm == 1 else "H",
                    "thin" if _thick == 1 else "thick",
                    "Ne"   if _core  == 1 else "CO",
                    bool(_mid),
                ))
        if show_timings:
            timings['[1] choose_tuples'] += time.perf_counter() - t_choose


    # --- PHASE 2: Pre-load all models and cache the interpolators actually used at query time ----------------------------------
    # ---------------------------------------------------------------------------------------------------------------------------
    if show_timings:
        t_cache = time.perf_counter()
        print(f"\nFound {len(all_unique_tuples)} unique model tuples. Pre-loading models...")
    for tpl in all_unique_tuples:
        loader.get_interpolator(tpl, output_key='logteff')               # He teff + two-step fallback
        loader.get_interpolator(tpl, output_key='Mag',   y_axis='age_cool')  # direct G
        loader.get_interpolator(tpl, output_key='color', y_axis='age_cool')  # direct bprp
    loader.build_distil_cache()
    loader.build_distil_atm_cache()
    if p.get("c_enrichment_prescription") is not None:
        _sdq_root = p.get(
            "stealth_dq_grid_root",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'White Dwarf Models', 'Camisassa StealthDQ'),
        )
        loader.build_sdq_cache(_sdq_root)
    cooling_source = p.get("cooling_source", "bedard").lower()
    if cooling_source == "camisassa":
        _cam_root = p.get(
            "camisassa_grid_root",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'White Dwarf Models', 'Camisassa'),
        )
        loader.build_camisassa_cache(_cam_root)
    if show_timings:
        timings['[2] model_cache_building'] = time.perf_counter() - t_cache
        print(f"Model cache built in {timings['[2] model_cache_building']:.3f}s\n")
    

    # --- PHASE 3: Process all groups with cached models ---------------------------------
    # -----------------------------------------------------------------------------------
    loop_iter = -1
    for mask_index, name, groups in mask_groups:
        loop_iter += 1
        group_items = list(groups.items())
        for i_iter, (tpl, group_idxs) in enumerate(group_items):
            group_idxs = np.array(group_idxs, dtype=int)

            masses_grp = all_data.loc[group_idxs, "Mass"].values
            ages_grp   = all_data.loc[group_idxs, "cool_age"].values  # Gyr expected

            # Try direct approach if enabled
            if use_direct_approach:
                # TRY APPROACH A: Direct (mass, age_cool) -> Mag/color
                mag_fn_direct = loader.get_interpolator(tpl, output_key='Mag', y_axis='age_cool')
                col_fn_direct = loader.get_interpolator(tpl, output_key='color', y_axis='age_cool')

                # Camisassa override: route non-Ne groups through the bundled
                # cooling+photometry grid when the user selected it. Ne-core stars
                # (mask name 'ne_nonmerger') keep their ONe tracks.
                use_cam = (cooling_source == "camisassa"
                           and name != "ne_nonmerger"
                           and loader.camisassa_grid is not None
                           and loader.camisassa_grid.supports(tpl[3]))
                if show_timings:
                    t_direct = time.perf_counter()
                    _t1 = _t2 = _t3 = _t4 = 0.0

                if use_cam:
                    mag_vals_direct, col_vals_direct = loader.camisassa_grid.get_photometry(
                        masses_grp, ages_grp, tpl[3])
                    # Fall back to the tuple's Bedard interpolators for stars that
                    # fell outside the Camisassa mass/age coverage.
                    miss = np.isnan(mag_vals_direct) | np.isnan(col_vals_direct)
                    if miss.any():
                        mag_vals_direct[miss] = mag_fn_direct(masses_grp[miss], ages_grp[miss])
                        col_vals_direct[miss] = col_fn_direct(masses_grp[miss], ages_grp[miss])
                    if show_timings:
                        _t1 = time.perf_counter() - t_direct; t_direct = time.perf_counter()
                else:
                    mag_vals_direct = mag_fn_direct(masses_grp, ages_grp)
                    if show_timings:
                        _t1 = time.perf_counter() - t_direct; t_direct = time.perf_counter()
                    col_vals_direct = col_fn_direct(masses_grp, ages_grp)
                    if show_timings:
                        _t2 = time.perf_counter() - t_direct; t_direct = time.perf_counter()

                all_data.loc[group_idxs, "G"] = mag_vals_direct
                if show_timings:
                    _t3 = time.perf_counter() - t_direct; t_direct = time.perf_counter()
                all_data.loc[group_idxs, "bprp"] = col_vals_direct
                if show_timings:
                    _t4 = time.perf_counter() - t_direct
                    _total = _t1 + _t2 + _t3 + _t4
                    suffix = " [cam]" if use_cam else ""
                    name_direct = f" [3.1.{loop_iter}.{i_iter}] {tpl} N={len(masses_grp)}{suffix}"
                    timings[name_direct] = _total
                    timings[f"  [3.1.{loop_iter}.{i_iter}.0] interp_mag"]  = _t1
                    timings[f"  [3.1.{loop_iter}.{i_iter}.1] interp_col"]  = _t2
                    timings[f"  [3.1.{loop_iter}.{i_iter}.2] assign_G"]    = _t3
                    timings[f"  [3.1.{loop_iter}.{i_iter}.3] assign_bprp"] = _t4
                    timings['[3] photometry'] += _total

            # Use two-step if not using direct approach
            if not use_direct_approach:
                if show_timings:
                    t_teff = time.perf_counter()
                
                # Get standard two-step interpolators
                teff_fn = loader.get_interpolator(tpl, output_key='logteff')                    # (mass, age) -> logteff
                mag_fn  = loader.get_interpolator(tpl, output_key='Mag',    y_axis='logteff')  # (mass, logteff) -> Mag
                col_fn  = loader.get_interpolator(tpl, output_key='color',  y_axis='logteff')  # (mass, logteff) -> color
                
                logteff_vals = None
                if teff_fn is not None:
                    try:
                        t_temp = time.perf_counter()
                        logteff_vals = teff_fn(masses_grp, ages_grp)
                        if show_timings:
                            name_teff = f" [3.0.{loop_iter}.{i_iter}] {tpl} (teff)"
                            timings[name_teff] = time.perf_counter() - t_temp
                        all_data.loc[group_idxs, "teff"] = 10 ** logteff_vals
                    except Exception:
                        all_data.loc[group_idxs, "teff"] = np.nan
                        logteff_vals = None
                
                if show_timings:
                    timings['[3.0] interp_teff'] += time.perf_counter() - t_teff

                # Compute photometry from teff
                if show_timings:
                    t_ph = time.perf_counter()
                if logteff_vals is not None:
                    # Filter for valid teff values
                    valid_mask = ~np.isnan(logteff_vals)
                    if valid_mask.any():
                        valid_idxs = group_idxs[valid_mask]
                        valid_masses = masses_grp[valid_mask]
                        valid_logteff = logteff_vals[valid_mask]
                        
                        # Compute both magnitude and color in one timing block
                        if mag_fn is not None:
                            try:
                                all_data.loc[valid_idxs, "G"] = mag_fn(valid_masses, valid_logteff)
                            except Exception:
                                all_data.loc[valid_idxs, "G"] = np.nan
                        
                        if col_fn is not None:
                            try:
                                all_data.loc[valid_idxs, "bprp"] = col_fn(valid_masses, valid_logteff)
                            except Exception:
                                all_data.loc[valid_idxs, "bprp"] = np.nan
                
                if show_timings:
                    name_phot = f" [3.1.{loop_iter}.{i_iter}] {tpl} (phot)"
                    timings[name_phot] = time.perf_counter() - t_ph
                    timings['[3] photometry'] += time.perf_counter() - t_ph

            # progress callback update
            if progress_callback:
                try:
                    frac_local = (i_iter + 1) / max(1, len(group_items))
                    frac_total = mask_index * per_mask_weight + frac_local * per_mask_weight
                    progress_callback(min(0.99, frac_total), f"{name}: {i_iter+1}/{len(group_items)} groups")
                except Exception:
                    pass


    # --- PHASE 4: distilled targets - assign teff via external distilled grid (vectorized) -----------------------
    # -------------------------------------------------------------------------------------------------------------
    idxs_distil = np.where(~mask_nd)[0]
    if show_timings:
        t0_distil = t_c = time.perf_counter()
    if idxs_distil.size:
        all_data.loc[idxs_distil, 'teff'] = loader.get_distil_teff(
            all_data.loc[idxs_distil, "Mass"].values,
            all_data.loc[idxs_distil, "cool_age"].values,
        )
        if show_timings:
            timings[' [4.0] distilled_teff'] = time.perf_counter() - t_c
            timings[' [4.2] distilled_mag'] = 0.0
            timings[' [4.3] distilled_color'] = 0.0
            t_c = time.perf_counter()
        # attempt photometry for distilled targets using loader (best-effort)
        # Use same efficient tuple selection as non-distilled section
        atm_types_d = all_data["atm_type_flag"].iloc[idxs_distil].values
        thicknesses_d = all_data["atm_thickness_flag"].iloc[idxs_distil].values
        core_types_d = all_data["core_flag"].iloc[idxs_distil].values
        mass_in_range = (all_data["Mass"].iloc[idxs_distil].values >= mid_low) & (all_data["Mass"].iloc[idxs_distil].values < mid_high)
        
        combinations_d = np.column_stack([atm_types_d, thicknesses_d, core_types_d, mass_in_range.astype(int)])  # shape (N, 4) - include midrange mass as a factor
        unique_rows_d, inverse_d = np.unique(combinations_d, axis=0, return_inverse=True)
        
        groups_d = defaultdict(list)
        for i, unique_row in enumerate(unique_rows_d):
            atm, thick, core, mass_is_midrange = unique_row
            atm = "He" if atm == 1 else "H"
            thick = "thin" if thick == 1 else "thick"
            core = "Ne" if core == 1 else "CO"
            tpl = loader.choose_tuple(atm=atm, thickness=thick, core=core, mass_is_midrange=bool(mass_is_midrange))
            group_idxs = idxs_distil[inverse_d == i]
            if len(group_idxs) > 0:
                groups_d[tpl].extend(group_idxs.tolist())
        
        group_items_d = list(groups_d.items())
        for i_iter, (tpl, group_idxs) in enumerate(group_items_d):
            group_idxs = np.array(group_idxs, dtype=int)
            atm_type_str = tpl[3]  # "H" or "He"
            teff_vals = all_data.loc[group_idxs, "teff"].values
            valid = (~np.isnan(teff_vals)) & (teff_vals > 0)
            sel = group_idxs[valid]
            if len(sel) == 0:
                continue
            logteff_sel = np.log10(all_data.loc[sel, "teff"].values)
            masses_sel  = all_data.loc[sel, "Mass"].values
            try:
                if show_timings:
                    t_ph = time.perf_counter()
                G_vals, bprp_vals = loader.get_distil_photometry(masses_sel, logteff_sel, atm_type_str)
                all_data.loc[sel, "G"] = G_vals
                if show_timings:
                    timings[' [4.2] distilled_mag'] += time.perf_counter() - t_ph
                    t_ph = time.perf_counter()
                all_data.loc[sel, "bprp"] = bprp_vals
                if show_timings:
                    timings[' [4.3] distilled_color'] += time.perf_counter() - t_ph
            except Exception:
                pass

            # progress callback update for distilled
            if progress_callback:
                try:
                    frac_local = (i_iter + 1) / max(1, len(group_items_d))
                    frac_total = weights_nd + frac_local * 0.2
                    progress_callback(min(0.99, frac_total), f"distilled: {i_iter+1}/{len(group_items_d)} groups")
                except Exception:
                    pass

    if show_timings:
        timings['[4] distilled'] = time.perf_counter() - t0_distil


    # --- PHASE 5: C-enrichment for He-rich WDs (Camisassa et al. 2023, stealth DQ) ----------------------
    # ----------------------------------------------------------------------------------------------------
    if show_timings:
        t0 = t_c = time.perf_counter()
    c_prescription = p.get("c_enrichment_prescription", None)
    if c_prescription is not None:
        _c_model = _CEnrichmentModel()
        sdq_grid = loader.sdq_grid
        he_mask = all_data["atm_type_flag"] == 1
        _c_seed = p.get("seed")
        _rng_c = np.random.default_rng(None if _c_seed is None else _c_seed + 1)
        # Per-star probability that a DB WD is C-enriched (routed through stealth DQ).
        # The complement stays on pure-He cooling tracks. Default 1.0 reproduces the
        # legacy behaviour (every DB gets the prescription). Applies in both named and
        # random prescription modes.
        _is_random = c_prescription.startswith('random')
        c_enriched_fraction = float(np.clip(p.get("c_enriched_fraction", 1.0), 0.0, 1.0))
        if show_timings:
            timings[' [5.1] sdq_init'] = time.perf_counter() - t_c
            t_c = time.perf_counter()

        _he_nd_idxs = np.where(he_mask & mask_nd)[0]
        if len(_he_nd_idxs) > 0:
            n_he      = len(_he_nd_idxs)
            masses_he = all_data["Mass"].iloc[_he_nd_idxs].values
            ages_he   = all_data["cool_age"].iloc[_he_nd_idxs].values
            core_he   = all_data["core_flag"].iloc[_he_nd_idxs].values

            # ---- Per-star routing ----
            # Ne-core He stars stay on their ONe tuple (stealth-DQ tables are CO physics).
            is_co       = (core_he != 1)
            is_enriched = (_rng_c.uniform(size=n_he) < c_enriched_fraction)
            if _is_random:
                _parts  = c_prescription.split('_')
                _lo     = float(_parts[1]) if len(_parts) > 1 else 0.0
                _hi     = float(_parts[2]) if len(_parts) > 2 else 2.0
                offsets = _rng_c.uniform(_lo, _hi, size=n_he)
            else:
                offsets = None

            is_c_enriched = is_co & is_enriched

            # ---- C-enriched stars: (logteff, G, bprp) straight from stealth-DQ ----
            if is_c_enriched.any():
                c_local  = np.where(is_c_enriched)[0]
                c_global = _he_nd_idxs[c_local]
                _m_c = masses_he[c_local]
                _a_c = ages_he[c_local]

                if _is_random:
                    _lt_c, _G_c, _bp_c = sdq_grid.get_observables_offset(
                        _m_c, _a_c, offsets[c_local])
                else:
                    _lt_c, _G_c, _bp_c = sdq_grid.get_observables(
                        _m_c, _a_c, c_prescription)

                # Stars that fell outside stealth-DQ coverage drop back to the pure-He
                # path below (camisassa/Bedard). c_he_log stays at PURE_HE_LIMIT for them.
                _miss = np.isnan(_lt_c) | np.isnan(_G_c) | np.isnan(_bp_c)
                if _miss.any():
                    is_c_enriched[c_local[_miss]] = False
                _ok = ~_miss
                if _ok.any():
                    ok_idx   = c_global[_ok]
                    _teff_ok = 10 ** _lt_c[_ok]
                    all_data.loc[ok_idx, "teff"] = _teff_ok
                    all_data.loc[ok_idx, "G"]    = _G_c[_ok]
                    all_data.loc[ok_idx, "bprp"] = _bp_c[_ok]
                    if _is_random:
                        _base = _c_model._base_c_sequence(_teff_ok)
                        _ch   = np.maximum(_base - offsets[c_local[_ok]],
                                           _c_model.PURE_HE_LIMIT)
                    else:
                        _ch = _c_model.get_c_he(_teff_ok, prescription=c_prescription)
                    all_data.loc[ok_idx, "c_he_log"] = _ch

            if show_timings:
                timings[' [5.2] sdq_observables'] = time.perf_counter() - t_c
                t_c = time.perf_counter()

            # ---- Pure-He stars (Ne-core, no-C, or stealth-DQ OOB): camisassa / Bedard ----
            is_pure_he = ~is_c_enriched
            if is_pure_he.any():
                ph_local  = np.where(is_pure_he)[0]
                ph_global = _he_nd_idxs[ph_local]
                ph_masses = masses_he[ph_local]
                ph_ages   = ages_he[ph_local]

                _mid_lo, _mid_hi = p.get("mid_mass_range", (0.9, 1.1))
                _combos = np.column_stack([
                    all_data["atm_type_flag"].iloc[ph_global].values,
                    all_data["atm_thickness_flag"].iloc[ph_global].values,
                    all_data["core_flag"].iloc[ph_global].values,
                    ((ph_masses >= _mid_lo) & (ph_masses < _mid_hi)).astype(int),
                ])
                _unique_rows, _inverse = np.unique(_combos, axis=0, return_inverse=True)
                for _i, _row in enumerate(_unique_rows):
                    _atm, _thick, _core, _midrange = _row
                    _tpl = loader.choose_tuple(
                        "He" if _atm == 1 else "H",
                        "thin" if _thick == 1 else "thick",
                        "Ne" if _core == 1 else "CO",
                        bool(_midrange),
                    )
                    _sel    = (_inverse == _i)
                    _idxs   = ph_global[_sel]
                    _masses = ph_masses[_sel]
                    _ages   = ph_ages[_sel]

                    _use_cam_he = (cooling_source == "camisassa"
                                   and _core != 1
                                   and loader.camisassa_grid is not None
                                   and loader.camisassa_grid.supports("He"))
                    if _use_cam_he:
                        _g, _bp, _lt = loader.camisassa_grid.get_all(_masses, _ages, "He")
                        _miss_lt = np.isnan(_lt)
                        if _miss_lt.any():
                            _fn_lt = loader.get_interpolator(_tpl, "logteff")
                            if _fn_lt is not None:
                                _lt[_miss_lt] = _fn_lt(_masses[_miss_lt], _ages[_miss_lt])
                        _miss_g = np.isnan(_g)
                        if _miss_g.any():
                            _fn_mag = loader.get_interpolator(_tpl, "Mag", y_axis="age_cool")
                            if _fn_mag is not None:
                                _g[_miss_g] = _fn_mag(_masses[_miss_g], _ages[_miss_g])
                        _miss_bp = np.isnan(_bp)
                        if _miss_bp.any():
                            _fn_col = loader.get_interpolator(_tpl, "color", y_axis="age_cool")
                            if _fn_col is not None:
                                _bp[_miss_bp] = _fn_col(_masses[_miss_bp], _ages[_miss_bp])
                    else:
                        _fn_lt  = loader.get_interpolator(_tpl, "logteff")
                        _fn_mag = loader.get_interpolator(_tpl, "Mag",   y_axis="age_cool")
                        _fn_col = loader.get_interpolator(_tpl, "color", y_axis="age_cool")
                        _lt = _fn_lt(_masses, _ages)  if _fn_lt  is not None else np.full(len(_masses), np.nan)
                        _g  = _fn_mag(_masses, _ages) if _fn_mag is not None else np.full(len(_masses), np.nan)
                        _bp = _fn_col(_masses, _ages) if _fn_col is not None else np.full(len(_masses), np.nan)

                    _ok_lt = ~np.isnan(_lt)
                    if _ok_lt.any():
                        all_data.loc[_idxs[_ok_lt], "teff"] = 10 ** _lt[_ok_lt]
                    _ok_g = ~np.isnan(_g)
                    if _ok_g.any():
                        all_data.loc[_idxs[_ok_g], "G"] = _g[_ok_g]
                    _ok_bp = ~np.isnan(_bp)
                    if _ok_bp.any():
                        all_data.loc[_idxs[_ok_bp], "bprp"] = _bp[_ok_bp]

                # Pure-He cooling tracks carry no C: c_he_log saturates at PURE_HE_LIMIT.
                all_data.loc[ph_global, "c_he_log"] = _c_model.PURE_HE_LIMIT

            if show_timings:
                timings[' [5.3] pure_he_path'] = time.perf_counter() - t_c
    if show_timings:
        timings['[5] c_enrichment'] = time.perf_counter() - t0


    # --- PHASE 6: Add companion WD flux for non-merged binaries ----------------------
    # ---------------------------------------------------------------------------------
    if show_timings:
        t0 = time.perf_counter()
    # Companion WD cooling age is assumed equal to the primary's (common-envelope assumption).
    if "companion_mass" in all_data.columns:
        nb_mask = (all_data["companion_mass"] > 0) & (all_data["merger_flag"] == 0) & (~np.isnan(all_data["G"]))
        if nb_mask.any():
            if show_timings:
                t_c = time.perf_counter()

            comp_df = all_data.loc[nb_mask, ["companion_mass", "cool_age",
                                              "atm_type_flag", "atm_thickness_flag"]].copy()
            comp_df = comp_df.rename(columns={"companion_mass": "Mass"})
            comp_df["rounded_Mass"] = np.round(comp_df["Mass"] / 0.05) * 0.05
            comp_df["core_flag"] = 0
            comp_df["merger_flag"] = 0
            comp_df["distil_flag"] = 0
            comp_df["companion_mass"] = 0.0
            comp_df["weight"] = 1.0
            comp_df["atm_type"] = np.where(comp_df["atm_type_flag"] == 0, "H", "He")
            comp_df["atm_thickness"] = np.where(comp_df["atm_thickness_flag"] == 0, "thick", "thin")
            comp_df["core_type"] = "CO"
            comp_df["G"] = np.nan
            comp_df["bprp"] = np.nan
            comp_df["teff"] = np.nan
            comp_df = comp_df.reset_index(drop=True)
            if show_timings:
                timings[' [6.1] comp_df_build'] = time.perf_counter() - t_c
                t_c = time.perf_counter()

            comp_df, _ = extract_observables(comp_df, params, loader=loader,
                                              use_direct_approach=use_direct_approach)
            if show_timings:
                timings[' [6.2] extract_obs'] = time.perf_counter() - t_c
                t_c = time.perf_counter()

            comp_G    = comp_df["G"].values
            comp_bprp = comp_df["bprp"].values
            wd_G    = all_data.loc[nb_mask, "G"].values
            wd_bprp = all_data.loc[nb_mask, "bprp"].values
            valid = ~np.isnan(comp_G) & ~np.isnan(wd_G)
            if valid.any():
                idx = np.where(nb_mask)[0][valid]
                f_wd    = 10 ** (-wd_G[valid]   / 2.5)
                f_comp  = 10 ** (-comp_G[valid] / 2.5)
                f_total = f_wd + f_comp
                all_data.iloc[idx, all_data.columns.get_loc("G")]    = -2.5 * np.log10(f_total)
                all_data.iloc[idx, all_data.columns.get_loc("bprp")] = (
                    wd_bprp[valid] * f_wd + comp_bprp[valid] * f_comp
                ) / f_total
            if show_timings:
                timings[' [6.3] flux_blend'] = time.perf_counter() - t_c
    if show_timings:
        timings['[6] binary_companion'] = time.perf_counter() - t0


    # --- PHASE 7: validity and selection: no distance assignment, select by absolute magnitude ----------------------
    # ----------------------------------------------------------------------------------------------------------------
    if show_timings:
        t0 = time.perf_counter()
    all_data["Valid"] = (~np.isnan(all_data["teff"]) | use_direct_approach) & (~np.isnan(all_data["G"])) & (~np.isnan(all_data["bprp"]))

    # absG is always the absolute G magnitude — plots read this column.
    # The selection model below may sample distances internally for filtering,
    # but it MUST NOT modify absG or bprp.
    all_data["absG"] = all_data["G"]

    sel_model = p.get("selection_model")
    if sel_model is None:
        sel_model = _AbsoluteMagCap(G_max=p.get("G_max", np.inf))
    elif not isinstance(sel_model, _SelectionModel):
        raise TypeError(
            "params['selection_model'] must be an instance of SelectionModel"
        )
    # --- Optional distance assignment (shared by selection + error model) --
    # Needed when the error model is on (errors scale with apparent mag, hence
    # distance) and/or a distance-aware selection model is chosen. Off by
    # default -> no distance, no *_obs columns, legacy output byte-identical.
    #
    # Distances are drawn UNIFORMLY IN VOLUME within volume_radius_pc:
    # p(d) ∝ d^2 via the inverse CDF d = R * u^(1/3). This is the most physical
    # choice for the ~120 pc solar neighbourhood: that radius sits well inside
    # the WD thin-disk vertical scale height (~250–300 pc), so the
    # exponential-disk density gradient is a sub-~10% effect across the sphere
    # and largely cancels — not worth the extra (data-unconstrained)
    # scale-height parameter at this distance. See
    # docs/programmatic/gaia_observational_errors.md.
    err_model = p.get("error_model")
    if err_model is not None and not isinstance(err_model, _ErrorModel):
        raise TypeError("params['error_model'] must be an instance of ErrorModel")
    needs_dist = (err_model is not None) or getattr(sel_model, "needs_distance", False)
    rng_obs = np.random.default_rng(p.get("seed"))
    if needs_dist:
        R = float(p.get("volume_radius_pc", 120.0))
        u = rng_obs.random(len(all_data))
        all_data["_sel_distance"] = R * np.cbrt(u)   # pc, uniform in volume

    selection_mask = all_data["Valid"].values & sel_model.apply(all_data, p)

    # --- Optional Gaia observational errors -------------------------------
    if err_model is not None:
        err_model.apply(all_data, p, rng=rng_obs)
    if show_timings:
        timings['[7] validity_selection'] = time.perf_counter() - t0

    if progress_callback:
        try:
            progress_callback(1.0, "Completed")
        except Exception:
            pass

    # Record total time and print summary (only if show_timings=True)
    if show_timings:
        timings['total'] = time.perf_counter() - t_total_start - timings['[2] model_cache_building']
        
        print("\n" + "="*70)
        print("EXTRACT_OBSERVABLES TIMING BREAKDOWN")
        print("="*70)
        print(f"{'Stage':<35} {'Time (s)':<12} {'Percentage':<10}")
        print("-"*70)
        sum_total, sum_pct = 0, 0
        def sort_key(k):
            if k.strip()[0] != '[':
                return (-1,)
            match = re.match(r'\[(\d+(?:\.\d+)*)\]', k.strip())
            if match:
                return tuple(int(x) for x in match.group(1).split('.'))
            return (-1,)
        
        sorted_stages = sorted(timings.keys(), key=sort_key)
        for stage in sorted_stages:
            if stage.strip()[0] != '[':
                continue
            if stage[:1] == " ":
                print(f"{stage:<35} {timings[stage]:<12.4f}")
                continue
            if stage !='[2] model_cache_building':
                sum_total += timings[stage]
                pct = (timings[stage] / timings['total'] * 100) if timings['total'] > 0 else 0
                sum_pct += pct
            print(f"{stage:<35} {timings[stage]:<12.4f} {pct:<10.2f}%")
        
        print("-"*70)
        print(f"{'SUM':<35} {sum_total:<12.4f} {sum_pct:<10.2f}%")
        print(f"{'TOTAL':<35} {timings['total']:<12.4f} {'100.00':<10}%")
        print("="*70 + "\n")

    return all_data, selection_mask


class DefaultParameters:
    """Class to hold default simulation parameters."""
    defaults = {
        "IMF_exponent": -2.3,
        "ms_mass_range": [0.5, 8.0],
        "ms_lifetime_law": "powerlaw",       # MS lifetime: "powerlaw" (10*M^-2.5) or "hurley" (Z-dependent)
        "use_amr": False,                    # chemical-enrichment age-metallicity relation; forces "hurley" when True
        "amr_z_present": 0.02,               # Z assigned to the most-recent burst (solar); AMR normalises to this
        "amr_z_floor": 1e-3,                 # lower clamp on Z for the oldest bursts (above Hurley validity ~1e-4)
        "amr_scatter": 0.0,                  # optional per-star Gaussian scatter in log10(Z/0.02) [dex]
        "IFMR_parameters": [0.4, 0.15],  # M_final = 0.3 + 0.1 * M_initial
        "dwd_fraction": 0.05,                # fraction of all WDs that come from a DWD channel (Torres+22: ~1.2% unresolved + DWD-merger share)
        "wdsg_fraction": 0.20,               # fraction of all WDs that come from the WD+SG channel (Torres+22: ~6.3% unresolved + WDMS-merger share)
        "dwd_imf_alpha":  0.2,               # power-law exponent for DWD secondary mass: p(M_2) ∝ M_2^-alpha on [eps, M_primary] (0 = uniform)
        "wdsg_imf_alpha": 0.2,               # power-law exponent for WD+SG secondary mass: p(M_2) ∝ M_2^-alpha on [eps, M_primary] (0 = uniform)
        "wdsg_delay": 1e7,                   # yr — constant WD+SG merger delay
        "distil_probability": 0.07,
        "type_b_fraction": 0.30,
        "c_enrichment_prescription": "random_0_2",   # None disables; options: c_sequence/minus1/minus2/minus3/random_0_2/random_0_3
        "c_enriched_fraction": 1.0,                  # per-star prob. that a DB WD is C-enriched; the complement stays on pure-He cooling tracks
        "cooling_source": "camisassa",               # cooling+photometry provider: "bedard" (Bergeron) or "camisassa" (LPCODE H+He CO tracks)
        "atm_thin_fraction": 0.10,
        "ne_core_mass_threshold": 1.05,
        "ne_core_mass_threshold_cohe": 1.20, # M☉ — ONe threshold for CO+He DWD mergers (DA+non-DA)
        "ne_core_fraction": 0.20,            # DEPRECATED — no longer used; core assignment is deterministic
        "time_start": 0,
        "time_stop": 10.5e9,
        "time_step": 1e7,
        "sfr_type": "exponential",
        "burst_N": 200,
        "sfr_tau": 3e9,
        "use_importance_sampling": False,
        "extra_bump": {"t_center": 7.0e9, "sigma": 5e8, "amp_fraction": 0.4},
        "HR_grid": (-0.5, 2, 0.002, 9, 16, 0.01),
        "HR_bands": ('bp3-rp3', 'G3'),
        "G_max": 15.0,
        "error_model": None,                 # None disables Gaia errors; else an ErrorModel instance (see error_models.py)
        "volume_radius_pc": 120.0,           # solar-neighbourhood radius for uniform-in-volume distance assignment (errors only)
        "save_file": None,
        "seed": 42,
        "plot_grid": True,
    }

def simulate_HR_all(params=None):
    """High-level wrapper: create population and extract observables."""
    defaults = DefaultParameters.defaults  
    if params is None:
        params = {}
    p = {**defaults, **params}

    print("Creating Population")
    all_data = create_population(p, show_times=True)
    print(f"Created {len(all_data['Mass'])} WDs")

    loader = WD_models_loader_wrapper.WDModelsLoader(
        HR_grid=p["HR_grid"], HR_bands=p["HR_bands"],
        lazy=True, interp_method="grid", grid_res=(300, 800),
    )

    print("Extracting Observables")
    all_data, selection_mask = extract_observables(
        all_data, p, loader=loader, show_timings=True, use_direct_approach=True,
    )

    # existing plotting / saving / return behavior preserved
    mask = (all_data["Valid"] == 1) & selection_mask
    try:
        model_for_plot = loader.get_hr_contour_model(('be', 'be', 'o', 'H'))
    except Exception:
        model_for_plot = None
    ax = plot_HR(all_data["bprp"][mask] if "bprp" in all_data.columns else np.zeros(mask.sum()),
                 all_data["G"][mask] if "G" in all_data.columns else np.zeros(mask.sum()),
                 all_data["true_age"][mask], model=model_for_plot, grid=p["plot_grid"], alpha=0.5)
    plt.show()

    save_df = all_data.loc[selection_mask].copy()
    if "save_file" in p and p["save_file"] is not None:
        save_path = save_df_with_metadata(save_df, p["save_file"], metadata=p)
        print(f"Saved selection to {save_path}")

    return all_data, selection_mask


def main():
    parameters = DefaultParameters.defaults
    
    # Run comparison
    simulate_HR_all(parameters)
    print("\n\n[Ending Program]\n\n")

    
if __name__ == "__main__":
    main()

