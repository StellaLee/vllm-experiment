#!/usr/bin/env python3
"""3-trial replication of the widened-distribution/closed-loop (conc=20) fleet-scale check --
same procedure as cf_robustness_check_poisson_replicated.py, applied to the closed-loop arm.

Per-trial whale-window mean ramp rates (W/s, threshold=15000 chars, from
analyze_power_shape_whale.py on each trial's own files):
  mono (b16384):  t1=50.4  t2=52.9  t3=49.6
  chunk=512:      t1=29.1  t2=36.6  t3=31.0
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate
from ramp_reserve_procurement import collect_peak_ramps, reserve_levels

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
T_window, dt, warmup = 100.0, 0.1, 50.0

TRIALS = [
    dict(t=1, rec=f"{RAW}/2026-07-25-pesim_gate_wide-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-25-pesim_gate_wide-b16384-t1-power.csv",
         mono_ramp=50.4, chunk_ramp=29.1),
    dict(t=2, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2-power.csv",
         mono_ramp=52.9, chunk_ramp=36.6),
    dict(t=3, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3-power.csv",
         mono_ramp=49.6, chunk_ramp=31.0),
]


def main():
    reductions = {"95%": [], "99%": [], "99.9%": []}
    single_server_reductions = []

    for tr in TRIALS:
        label = f"trial {tr['t']}"
        print(f"\n{'='*60}\n{label}\n{'='*60}")
        cal = calibrate(tr["rec"], tr["pw"], label, whale_thresh=15000)
        tau_mono = (cal["P_max"] - cal["P_b"]) / tr["mono_ramp"]
        tau_chunk = (cal["P_max"] - cal["P_b"]) / tr["chunk_ramp"]
        single_reduction = (1 - tr["chunk_ramp"] / tr["mono_ramp"]) * 100
        single_server_reductions.append(single_reduction)
        print(f"  single-server ramp reduction: {single_reduction:.1f}% "
              f"({tr['mono_ramp']}->{tr['chunk_ramp']} W/s)")

        N, n_mc = 10000, 300
        rng = np.random.default_rng(20260807 + tr["t"])
        mono_peaks = collect_peak_ramps(N, 0.0, cal, tau_mono, T_window, dt, n_mc, rng, warmup)
        chunk_peaks = collect_peak_ramps(N, 0.0, cal, tau_chunk, T_window, dt, n_mc, rng, warmup)
        mono_r = reserve_levels(mono_peaks, f"{label}: mono")
        chunk_r = reserve_levels(chunk_peaks, f"{label}: chunk=512")
        for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
            m, c = mono_r[key], chunk_r[key]
            red = (1 - c / m) * 100
            reductions[level].append(red)
            print(f"    {level:>6}: mono={m:.4f}  chunk=512={c:.4f} MW/min  reduction={red:.1f}%")

    print(f"\n{'='*60}\nSUMMARY ACROSS 3 TRIALS (mean +/- std)\n{'='*60}")
    ssr = np.array(single_server_reductions)
    print(f"single-server ramp reduction: {ssr.mean():.1f}% +/- {ssr.std():.1f}  "
          f"(per-trial: {[f'{x:.1f}' for x in ssr]})")
    for level, vals in reductions.items():
        v = np.array(vals)
        print(f"reserve reduction {level:>6}: {v.mean():.1f}% +/- {v.std():.1f}  "
              f"(per-trial: {[f'{x:.1f}' for x in v]})")


if __name__ == "__main__":
    main()
