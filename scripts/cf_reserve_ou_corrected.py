#!/usr/bin/env python3
"""Redo of the reserve-procurement check (raw peak-ramp percentiles, matching
ramp_reserve_procurement.py's actual metric) using the CONTINUOUS OU-noise model instead of
the flat two-state model -- checking whether realistic within-regime noise changes the raw
reserve number, now that we know CF_ramp (used earlier this session) was the wrong proxy for
this question."""
import sys
import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous, ou_continuous_aggregate
from coincidence_factor_model import simulate_states

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
RNG = np.random.default_rng(20260807)


def collect_peak_ramps_ou(N, cal, kappa, T_window, dt, n_mc, rng, warmup=50.0):
    i0 = int(warmup / dt)
    peak_ramps = np.empty(n_mc)
    for k in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                     cal["sigma_max"], dt, kappa, rng)
        D = P.sum(axis=0)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps[k] = ramp[i0:].max() if len(ramp) > i0 else 0.0
    return peak_ramps


def reserve_levels(peak_ramps_w_per_s, label):
    mw_per_min = peak_ramps_w_per_s * 60.0 / 1e6
    emp_95, emp_99 = np.percentile(mw_per_min, 95), np.percentile(mw_per_min, 99)
    loc, scale = gumbel_r.fit(mw_per_min)
    fit_999 = gumbel_r.ppf(0.999, loc=loc, scale=scale)
    print(f"[{label}] mean={mw_per_min.mean():.4f} P95={emp_95:.4f} P99={emp_99:.4f} "
          f"P99.9(Gumbel)={fit_999:.4f} MW/min")
    return dict(R95=emp_95, R99=emp_99, R999=fit_999)


def main():
    N = 10000
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc = 200  # OU sim is the slow one (sequential recurrence); fewer reps to stay fast

    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv", "mono")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv", "chunk=512")
    # kappa values already fitted earlier this session (mean|dP/dt| matches 45.7 / 30.8 W/s)
    kappa_mono, kappa_chunk = 0.1320, 0.0710

    mono_peaks = collect_peak_ramps_ou(N, cal_mono, kappa_mono, T_window, dt, n_mc, RNG, warmup)
    print(f"mono done")
    chunk_peaks = collect_peak_ramps_ou(N, cal_chunk, kappa_chunk, T_window, dt, n_mc, RNG, warmup)
    print(f"chunk done\n")

    mono_r = reserve_levels(mono_peaks, "mono (OU noise)")
    chunk_r = reserve_levels(chunk_peaks, "chunk=512 (OU noise)")

    print("\n=== RESERVE PROCUREMENT, CONTINUOUS OU MODEL (N=10,000) ===")
    for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
        m, c = mono_r[key], chunk_r[key]
        reduction = (1 - c / m) * 100
        print(f"  {level:>6} reliability: mono={m:.4f} MW/min  chunk=512={c:.4f} MW/min  "
              f"reduction={reduction:.1f}%")


if __name__ == "__main__":
    main()
