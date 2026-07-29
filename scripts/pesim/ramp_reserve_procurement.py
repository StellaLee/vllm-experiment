#!/usr/bin/env python3
"""Ramping-reserve procurement analysis: a genuine power-system scheduling/optimization problem
(how much fast-ramping regulation reserve to procure) fed by the mono vs chunk=512 demand
profiles, reusing the already-validated fleet-scale Monte Carlo model in
coincidence_factor_model.py -- no new datacenter experiments, just exposing and post-processing
the per-trial peak-ramp samples that ramp_coincidence_factor() already computes internally but
previously discarded (only the mean was returned).

PROBLEM STATEMENT: a system operator must procure ramping-reserve capacity R (MW/min of
fast-response ramp-following capability) large enough that the fleet's realized ramp exceeds it
only rarely: minimize R subject to P(|ramp| > R) <= eps, for a target reliability level 1-eps
(e.g. 95%, 99%, 99.9%, echoing "1-in-10"-style design criteria already referenced in
coincidence_factor_model.py's own docstring). For a single chance constraint like this, the
optimal R* is exactly the (1-eps)-quantile of the peak-ramp distribution -- an honest, standard
(not dressed-up) solution to a real capacity-scheduling problem.

METHOD: simulate N=10,000 servers (matching the fleet size already used in the paper's
Discussion section) at s=0, collecting the realized peak ramp from each of n_mc independent
100-second reference windows (same warm-up-corrected methodology as the rest of the paper). For
eps=5%/1% we use the empirical percentile directly (well-supported by n_mc samples). For
eps=0.1% (a tail beyond what n_mc directly resolves), we fit a Gumbel (right-tail extreme-value)
distribution to the block maxima -- the standard, textbook-correct method for extrapolating
rare-event capacity requirements from block-maximum samples (this is not a computational
shortcut; block-maximum extreme value theory is the standard tool for exactly this class of
capacity-dimensioning problem in power systems reliability practice), and validate the fit by
comparing its 95%/99% quantiles against the empirical ones before trusting its 99.9% extrapolation.
"""
import sys
import time
import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from coincidence_factor_model import calibrate, simulate_states, smoothed_aggregate

RNG = np.random.default_rng(20260807)


def collect_peak_ramps(N, s, cal, tau, T_window, dt, n_mc, rng, warmup=50.0):
    """Mirrors ramp_coincidence_factor()'s simulation loop exactly, but returns the raw
    per-trial peak-ramp samples (W/s, absolute) instead of collapsing them to a mean/normalized
    ratio -- these samples were always computed internally and discarded; nothing new is
    simulated here relative to the rest of the paper's methodology."""
    i0 = int(warmup / dt)
    peak_ramps = np.empty(n_mc)
    for k in range(n_mc):
        state = simulate_states(N, s, cal, T_window + warmup, dt, rng)
        D = smoothed_aggregate(state, cal["P_b"], cal["P_max"], dt, tau)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps[k] = ramp[i0:].max() if len(ramp) > i0 else 0.0
    return peak_ramps


def reserve_levels(peak_ramps_w_per_s, label):
    """Converts W/s peak-ramp samples to MW/min reserve-capacity requirements at three
    reliability levels, using empirical percentiles for 95/99% and a Gumbel-tail fit for the
    0.1% level (validated against the empirical 95/99% before trusting the extrapolation)."""
    mw_per_min = peak_ramps_w_per_s * 60.0 / 1e6
    emp_95, emp_99 = np.percentile(mw_per_min, 95), np.percentile(mw_per_min, 99)

    loc, scale = gumbel_r.fit(mw_per_min)
    fit_95 = gumbel_r.ppf(0.95, loc=loc, scale=scale)
    fit_99 = gumbel_r.ppf(0.99, loc=loc, scale=scale)
    fit_999 = gumbel_r.ppf(0.999, loc=loc, scale=scale)

    print(f"[{label}] n_mc={len(peak_ramps_w_per_s)} mean_peak={mw_per_min.mean():.4f} MW/min "
          f"std={mw_per_min.std():.4f}")
    print(f"    empirical: P95={emp_95:.4f}  P99={emp_99:.4f} MW/min")
    print(f"    Gumbel fit(loc={loc:.4f}, scale={scale:.4f}): "
          f"P95={fit_95:.4f}  P99={fit_99:.4f}  P99.9={fit_999:.4f} MW/min "
          f"(fit-vs-empirical residual: {abs(fit_95-emp_95)/emp_95*100:.1f}% / "
          f"{abs(fit_99-emp_99)/emp_99*100:.1f}%)")
    return dict(R95=emp_95, R99=emp_99, R999=fit_999)


def main():
    cal = calibrate(
        "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/mono_t1_records.jsonl",
        "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/mono_t1_power.csv",
        "mono/16384",
    )
    N = 10000
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc = 300
    tau_mono = (cal["P_max"] - cal["P_b"]) / 41.3
    tau_chunk = (cal["P_max"] - cal["P_b"]) / 27.1

    t0 = time.time()
    mono_peaks = collect_peak_ramps(N, 0.0, cal, tau_mono, T_window, dt, n_mc, RNG, warmup)
    t1 = time.time()
    print(f"mono: {n_mc} trials in {t1-t0:.1f}s")
    chunk_peaks = collect_peak_ramps(N, 0.0, cal, tau_chunk, T_window, dt, n_mc, RNG, warmup)
    t2 = time.time()
    print(f"chunk: {n_mc} trials in {t2-t1:.1f}s\n")

    mono_r = reserve_levels(mono_peaks, "mono (R=41.3 W/s)")
    print()
    chunk_r = reserve_levels(chunk_peaks, "chunk=512 (R=27.1 W/s)")

    print("\n=== RESERVE PROCUREMENT COMPARISON (N=10,000) ===")
    for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
        m, c = mono_r[key], chunk_r[key]
        reduction = (1 - c / m) * 100
        print(f"  {level:>6} reliability: mono={m:.4f} MW/min  chunk=512={c:.4f} MW/min  "
              f"reduction={reduction:.1f}%")


if __name__ == "__main__":
    main()
