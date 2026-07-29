#!/usr/bin/env python3
"""Redo of ramp_reserve_procurement.py's actual metric (raw peak-ramp percentiles -> MW/min
reserve requirement, NOT the CF_ramp-normalized ratio) using PER-POLICY corrected timing
(cal_mono, cal_chunk each from their own trace) instead of the original's shared-cal flaw.

This matters because CF_ramp normalizes each policy's aggregate peak by ITS OWN single-server
characteristic ramp before comparing -- which, by construction, divides away much of the very
effect the reserve-procurement question is actually about. The reserve pipeline never does that
normalization; it works with raw MW/min. So CF_ramp collapsing to ~0% earlier this session does
NOT by itself mean the reserve number collapses too -- this script checks directly."""
import sys
import time

import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from cf_continuous_regime_model import calibrate_continuous
from coincidence_factor_model import simulate_states, smoothed_aggregate

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
RNG = np.random.default_rng(20260807)


def collect_peak_ramps(N, cal, tau, T_window, dt, n_mc, rng, warmup=50.0):
    i0 = int(warmup / dt)
    peak_ramps = np.empty(n_mc)
    for k in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        D = smoothed_aggregate(state, cal["mu_b"], cal["mu_max"], dt, tau)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps[k] = ramp[i0:].max() if len(ramp) > i0 else 0.0
    return peak_ramps


def reserve_levels(peak_ramps_w_per_s, label):
    mw_per_min = peak_ramps_w_per_s * 60.0 / 1e6
    emp_95, emp_99 = np.percentile(mw_per_min, 95), np.percentile(mw_per_min, 99)
    loc, scale = gumbel_r.fit(mw_per_min)
    fit_999 = gumbel_r.ppf(0.999, loc=loc, scale=scale)
    print(f"[{label}] n_mc={len(peak_ramps_w_per_s)} mean_peak={mw_per_min.mean():.4f} MW/min "
          f"P95={emp_95:.4f} P99={emp_99:.4f} P99.9(Gumbel)={fit_999:.4f} MW/min")
    return dict(R95=emp_95, R99=emp_99, R999=fit_999)


def main():
    N = 10000
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc = 100

    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                                     "mono")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv",
                                      "chunk=512")
    tau_mono = (cal_mono["mu_max"] - cal_mono["mu_b"]) / 45.7
    tau_chunk = (cal_chunk["mu_max"] - cal_chunk["mu_b"]) / 30.8

    t0 = time.time()
    mono_peaks = collect_peak_ramps(N, cal_mono, tau_mono, T_window, dt, n_mc, RNG, warmup)
    t1 = time.time()
    print(f"mono: {n_mc} trials in {t1-t0:.1f}s")
    chunk_peaks = collect_peak_ramps(N, cal_chunk, tau_chunk, T_window, dt, n_mc, RNG, warmup)
    t2 = time.time()
    print(f"chunk: {n_mc} trials in {t2-t1:.1f}s\n")

    mono_r = reserve_levels(mono_peaks, "mono (per-policy timing)")
    chunk_r = reserve_levels(chunk_peaks, "chunk=512 (per-policy timing)")

    print("\n=== RESERVE PROCUREMENT, PER-POLICY CORRECTED TIMING (N=10,000) ===")
    for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
        m, c = mono_r[key], chunk_r[key]
        reduction = (1 - c / m) * 100
        print(f"  {level:>6} reliability: mono={m:.4f} MW/min  chunk=512={c:.4f} MW/min  "
              f"reduction={reduction:.1f}%")


if __name__ == "__main__":
    main()
