#!/usr/bin/env python3
"""Direct check: for the ORIGINAL (shared-timing) two-state model, does CF_ramp's reduction
% actually match the raw-percentile reduction %? Reconciling why the earlier session table
(two-state ~30% / OU noise ~0% / bootstrap ~35-53% worse) looked inconsistent with the later
corrected raw-percentile numbers (all ~25-29%)."""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate, simulate_states, smoothed_aggregate

RNG = np.random.default_rng(20260807)
RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def collect_peaks(N, cal, tau, T_window, dt, n_mc, rng, warmup=50.0):
    i0 = int(warmup / dt)
    peaks = np.empty(n_mc)
    for k in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        D = smoothed_aggregate(state, cal["P_b"], cal["P_max"], dt, tau)
        ramp = np.abs(np.diff(D)) / dt
        peaks[k] = ramp[i0:].max() if len(ramp) > i0 else 0.0
    return peaks


def main():
    # SAME shared cal for both mono and chunk -- this is the ORIGINAL (flawed) model exactly
    cal = calibrate(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                     "mono/16384 (shared cal)", whale_thresh=15000)
    tau_mono = (cal["P_max"] - cal["P_b"]) / 45.7
    tau_chunk = (cal["P_max"] - cal["P_b"]) / 30.8
    N = 10000
    T_window, dt, warmup, n_mc = 100.0, 0.1, 50.0, 60

    mono_peaks = collect_peaks(N, cal, tau_mono, T_window, dt, n_mc, RNG, warmup)
    chunk_peaks = collect_peaks(N, cal, tau_chunk, T_window, dt, n_mc, RNG, warmup)

    single_max_ramp_mono = (cal["P_max"] - cal["P_b"]) / tau_mono
    single_max_ramp_chunk = (cal["P_max"] - cal["P_b"]) / tau_chunk
    print(f"single_max_ramp: mono={single_max_ramp_mono:.2f} chunk={single_max_ramp_chunk:.2f} W/s")

    cf_mono = mono_peaks.mean() / (N * single_max_ramp_mono)
    cf_chunk = chunk_peaks.mean() / (N * single_max_ramp_chunk)
    print(f"\nCF_ramp: mono={cf_mono:.4f} chunk={cf_chunk:.4f}  "
          f"CF_ramp reduction={100*(1-cf_chunk/cf_mono):.1f}%")

    raw_mono, raw_chunk = mono_peaks.mean(), chunk_peaks.mean()
    print(f"\nRaw mean peak: mono={raw_mono:.2f} chunk={raw_chunk:.2f} W/s  "
          f"raw reduction={100*(1-raw_chunk/raw_mono):.1f}%")

    p95_mono, p95_chunk = np.percentile(mono_peaks, 95), np.percentile(chunk_peaks, 95)
    print(f"Raw P95 peak:  mono={p95_mono:.2f} chunk={p95_chunk:.2f} W/s  "
          f"P95 reduction={100*(1-p95_chunk/p95_mono):.1f}%")


if __name__ == "__main__":
    main()
