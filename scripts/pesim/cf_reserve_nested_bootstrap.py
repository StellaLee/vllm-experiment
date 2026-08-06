#!/usr/bin/env python3
"""Two-level (nested) bootstrap for the reserve-procurement Monte Carlo, fixing the honesty
gap found in the single-level "8 seeds" convention: the existing procedure fixes the trace
library and only re-randomizes phase/trace-assignment across virtual servers (Monte Carlo
noise given a fixed library), which the N-sweep and same-vs-different-trace correlation checks
showed does NOT shrink with N the way true independence would predict -- the effective degrees
of freedom is capped by the library size (n_traces), not N=10,000. That means "what if we'd
collected a different batch of real trials" is the dominant source of uncertainty, and the old
single-level std across seeds never measured it at all.

Outer loop (B replicates): resample the library itself -- draw n_traces trace-indices WITH
replacement from the available library (the standard nonparametric bootstrap unit: the REAL
TRIAL is the thing being resampled, not the virtual server). Inner loop (n_mc_inner reps): the
existing sample_fleet Monte Carlo (phase + trace-assignment across N=10,000 virtual servers) on
that resampled library. Report mean +/- std of R* and reduction % ACROSS THE OUTER REPLICATES --
this combines both real trace-selection uncertainty and within-library MC noise into one honest
number, rather than only the latter.
"""
import argparse
import glob
import sys

import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from cf_tracebootstrap_model import build_library, sample_fleet

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
TAG = "pesim_gate_convdiverse_conc10_gpu0"


def find_path(budget, trial):
    matches = glob.glob(f"{RAW}/*-{TAG}-{budget}-t{trial}-power.csv")
    assert len(matches) == 1, f"expected exactly 1 match for {budget} t{trial}, got {matches}"
    return matches[0]


def reserve_levels(peak_ramps_w_per_s):
    mw_per_min = peak_ramps_w_per_s * 60.0 / 1e6
    emp_95, emp_99 = np.percentile(mw_per_min, 95), np.percentile(mw_per_min, 99)
    loc, scale = gumbel_r.fit(mw_per_min)
    fit_999 = gumbel_r.ppf(0.999, loc=loc, scale=scale)
    return emp_95, emp_99, fit_999


def collect_peaks(lib, N, n_steps, dt, n_mc, rng):
    peaks = np.empty(n_mc)
    for k in range(n_mc):
        P = sample_fleet(lib, N, n_steps, 0.0, rng)
        peaks[k] = np.abs(np.diff(P.sum(axis=0))).max() / dt
    return peaks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=100)
    ap.add_argument("--n-outer", type=int, default=30, help="outer bootstrap replicates (resample the library)")
    ap.add_argument("--n-mc-inner", type=int, default=30, help="inner MC reps per outer replicate")
    ap.add_argument("--N", type=int, default=10000)
    args = ap.parse_args()

    dt = 0.1
    mono_paths = [find_path("b16384", i) for i in range(1, args.n_trials + 1)]
    chunk_paths = [find_path("b512", i) for i in range(1, args.n_trials + 1)]
    lib_mono = build_library(mono_paths, dt)
    lib_chunk = build_library(chunk_paths, dt)
    shortest = min(lib_mono.shape[1], lib_chunk.shape[1]) * dt
    n_steps = int(round(shortest * 0.5, -1) / dt)
    n_traces_m, n_traces_c = lib_mono.shape[0], lib_chunk.shape[0]
    print(f"n_trials={args.n_trials}  n_outer={args.n_outer}  n_mc_inner={args.n_mc_inner}  "
          f"window={n_steps*dt:.0f}s")

    rng = np.random.default_rng(20260731)
    r95m_l, r99m_l, r999m_l = [], [], []
    r95c_l, r99c_l, r999c_l = [], [], []
    red95_l, red99_l, red999_l = [], [], []

    for b in range(args.n_outer):
        idx_m = rng.integers(0, n_traces_m, size=n_traces_m)
        idx_c = rng.integers(0, n_traces_c, size=n_traces_c)
        resampled_mono = lib_mono[idx_m]
        resampled_chunk = lib_chunk[idx_c]

        peaks_m = collect_peaks(resampled_mono, args.N, n_steps, dt, args.n_mc_inner, rng)
        peaks_c = collect_peaks(resampled_chunk, args.N, n_steps, dt, args.n_mc_inner, rng)
        r95m, r99m, r999m = reserve_levels(peaks_m)
        r95c, r99c, r999c = reserve_levels(peaks_c)
        red95 = 100 * (1 - r95c / r95m)
        red99 = 100 * (1 - r99c / r99m)
        red999 = 100 * (1 - r999c / r999m)

        print(f"  outer={b:>2} n_unique_traces(mono/chunk)={len(set(idx_m.tolist()))}/{len(set(idx_c.tolist()))}  "
              f"mono R95/99/99.9={r95m:.3f}/{r99m:.3f}/{r999m:.3f}  chunk={r95c:.3f}/{r99c:.3f}/{r999c:.3f}  "
              f"reduction={red95:.1f}/{red99:.1f}/{red999:.1f}%")

        r95m_l.append(r95m); r99m_l.append(r99m); r999m_l.append(r999m)
        r95c_l.append(r95c); r99c_l.append(r99c); r999c_l.append(r999c)
        red95_l.append(red95); red99_l.append(red99); red999_l.append(red999)

    print()
    print(f"=== n_trials={args.n_trials}: TWO-LEVEL (nested) bootstrap, mean +/- std across {args.n_outer} outer replicates ===")
    for name, arr in [("95%", red95_l), ("99%", red99_l), ("99.9%", red999_l)]:
        arr = np.array(arr)
        print(f"  {name:>6}: reduction = {arr.mean():.1f}% +/- {arr.std():.1f}%")
    print()
    for name, m_arr, c_arr in [("95%", r95m_l, r95c_l), ("99%", r99m_l, r99c_l), ("99.9%", r999m_l, r999c_l)]:
        m_arr, c_arr = np.array(m_arr), np.array(c_arr)
        print(f"  {name:>6}: mono R*={m_arr.mean():.3f}+/-{m_arr.std():.3f}  chunk R*={c_arr.mean():.3f}+/-{c_arr.std():.3f} MW/min")


if __name__ == "__main__":
    main()
