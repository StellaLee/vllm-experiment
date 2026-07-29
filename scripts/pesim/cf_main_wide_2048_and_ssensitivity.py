#!/usr/bin/env python3
"""Two review-response add-ons to the main-body reserve-procurement result
(cf_main_wide_wholetrace.py, which produced Table I / tab:reserve using only mono vs
chunk=512):

(A) Same whole-trace methodology, chunk=2048 instead of chunk=512 -- chunk=2048 is closer to
    vLLM's actual production default and was dropped from the fleet-scale/reserve analysis
    after Section IV; a reviewer asked why, and whether its reserve number is also reported.
    Per-trial whole-trace chunk=2048 ramp rates (W/s, from Table II): t1=43.8 t2=43.4 t3=41.2.

(B) s-sensitivity: the main reserve result uses s=0 (fully independent arrivals) throughout,
    despite Section V's level-CF finding that correlated-arrival probability -- not diversity
    -- dominates aggregate PEAK risk. A reviewer asked whether the reserve number is an
    underestimate if arrivals are not perfectly independent. We re-run reserve_levels for
    mono and chunk=512 at s=0.02 and s=0.05 (trial-1 calibration only, N=10000) to report how
    much R95/R99 grow relative to the s=0 baseline already in the paper.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from coincidence_factor_model import calibrate
from ramp_reserve_procurement import collect_peak_ramps, reserve_levels

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
T_window, dt, warmup = 100.0, 0.1, 50.0

TRIALS = [
    dict(t=1, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
         mono_ramp=46.0, chunk_ramp=43.8),
    dict(t=2, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2-power.csv",
         mono_ramp=46.5, chunk_ramp=43.4),
    dict(t=3, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3-power.csv",
         mono_ramp=44.6, chunk_ramp=41.2),
]


def part_a():
    print("="*70)
    print("PART A: fleet-scale reserve, chunk=2048 (whole-trace R, same method as Table I)")
    print("="*70)
    reductions = {"95%": [], "99%": [], "99.9%": []}
    abs_levels = {"95%": [], "99%": [], "99.9%": []}
    ssr = []

    for tr in TRIALS:
        label = f"trial {tr['t']}"
        print(f"\n--- {label} ---")
        cal = calibrate(tr["rec"], tr["pw"], label, whale_thresh=15000)
        tau_mono = (cal["P_max"] - cal["P_b"]) / tr["mono_ramp"]
        tau_chunk = (cal["P_max"] - cal["P_b"]) / tr["chunk_ramp"]
        single_reduction = (1 - tr["chunk_ramp"] / tr["mono_ramp"]) * 100
        ssr.append(single_reduction)
        print(f"  single-server ramp reduction: {single_reduction:.1f}% "
              f"({tr['mono_ramp']}->{tr['chunk_ramp']} W/s)")

        N, n_mc = 10000, 300
        rng = np.random.default_rng(20260807 + tr["t"])
        mono_peaks = collect_peak_ramps(N, 0.0, cal, tau_mono, T_window, dt, n_mc, rng, warmup)
        chunk_peaks = collect_peak_ramps(N, 0.0, cal, tau_chunk, T_window, dt, n_mc, rng, warmup)
        mono_r = reserve_levels(mono_peaks, f"{label}: mono")
        chunk_r = reserve_levels(chunk_peaks, f"{label}: chunk=2048")
        for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
            m, c = mono_r[key], chunk_r[key]
            red = (1 - c / m) * 100
            reductions[level].append(red)
            abs_levels[level].append((m, c))
            print(f"    {level:>6}: mono={m:.4f}  chunk=2048={c:.4f} MW/min  reduction={red:.1f}%")

    print(f"\nSUMMARY (chunk=2048, 3 trials, mean +/- std)")
    ssr = np.array(ssr)
    print(f"single-server ramp reduction: {ssr.mean():.1f}% +/- {ssr.std():.1f}")
    for level, vals in reductions.items():
        v = np.array(vals)
        m_arr = np.array([a[0] for a in abs_levels[level]])
        c_arr = np.array([a[1] for a in abs_levels[level]])
        print(f"reserve reduction {level:>6}: {v.mean():.1f}% +/- {v.std():.1f}  "
              f"absolute mono={m_arr.mean():.4f} chunk2048={c_arr.mean():.4f} MW/min")


def part_b():
    print("\n" + "="*70)
    print("PART B: s-sensitivity of reserve requirement (trial-1 calibration, N=10000)")
    print("="*70)
    tr = TRIALS[0]
    cal = calibrate(tr["rec"], tr["pw"], "trial 1", whale_thresh=15000)
    tau_mono = (cal["P_max"] - cal["P_b"]) / tr["mono_ramp"]
    tau_chunk = (cal["P_max"] - cal["P_b"]) / 27.8  # chunk=512 trial-1 ramp, matches Table I calibration
    N, n_mc = 10000, 300

    for s in [0.0, 0.02, 0.05]:
        rng = np.random.default_rng(20260901)
        mono_peaks = collect_peak_ramps(N, s, cal, tau_mono, T_window, dt, n_mc, rng, warmup)
        chunk_peaks = collect_peak_ramps(N, s, cal, tau_chunk, T_window, dt, n_mc, rng, warmup)
        print(f"\n-- s={s} --")
        mono_r = reserve_levels(mono_peaks, f"mono (s={s})")
        chunk_r = reserve_levels(chunk_peaks, f"chunk=512 (s={s})")
        for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
            red = (1 - chunk_r[key] / mono_r[key]) * 100
            print(f"    {level}: mono={mono_r[key]:.4f} chunk=512={chunk_r[key]:.4f} "
                  f"MW/min  reduction={red:.1f}%")


if __name__ == "__main__":
    part_a()
    part_b()
