#!/usr/bin/env python3
"""Redo of the reserve-procurement check using the real-trace bootstrap, raw percentiles
(matching ramp_reserve_procurement.py's actual metric) instead of the CF_ramp-normalized ratio
used earlier this session -- checking whether the earlier "chunk is worse" sign flip survives
once measured the way the paper's reserve claim is actually computed."""
import sys
import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_tracebootstrap_model import build_library, sample_fleet, MONO_TRACES, CHUNK_TRACES

RNG = np.random.default_rng(20260807)


def collect_peak_ramps_bootstrap(library, N, dt, n_steps, n_mc, rng):
    peak_ramps = np.empty(n_mc)
    for k in range(n_mc):
        P = sample_fleet(library, N, n_steps, 0.0, rng)
        D = P.sum(axis=0)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps[k] = ramp.max()
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
    dt = 0.1
    N = 10000
    n_mc = 100

    lib_mono = build_library(MONO_TRACES, dt)
    lib_chunk = build_library(CHUNK_TRACES, dt)
    shortest = min(lib_mono.shape[1], lib_chunk.shape[1]) * dt
    T_sim = round(shortest * 0.5, -1)
    n_steps = int(T_sim / dt)
    print(f"T_sim={T_sim:.0f}s, n_mc={n_mc}, N={N}")

    mono_peaks = collect_peak_ramps_bootstrap(lib_mono, N, dt, n_steps, n_mc, RNG)
    chunk_peaks = collect_peak_ramps_bootstrap(lib_chunk, N, dt, n_steps, n_mc, RNG)

    mono_r = reserve_levels(mono_peaks, "mono (real-trace bootstrap)")
    chunk_r = reserve_levels(chunk_peaks, "chunk=512 (real-trace bootstrap)")

    print("\n=== RESERVE PROCUREMENT, REAL-TRACE BOOTSTRAP (N=10,000) ===")
    for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
        m, c = mono_r[key], chunk_r[key]
        reduction = (1 - c / m) * 100
        print(f"  {level:>6} reliability: mono={m:.4f} MW/min  chunk=512={c:.4f} MW/min  "
              f"reduction={reduction:.1f}%")


if __name__ == "__main__":
    main()
