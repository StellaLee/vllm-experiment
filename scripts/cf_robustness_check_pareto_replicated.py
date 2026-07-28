#!/usr/bin/env python3
"""3-trial replication of the Pareto-tail whale-size distribution/closed-loop (conc=20)
fleet-scale check -- same procedure as cf_robustness_check_wide_replicated.py /
cf_robustness_check_poisson_replicated.py, applied to the Pareto-tail arm (alpha=3.0,
xm=18000 chars, capped at 50000 chars; short-prompt body left at ORIGINAL narrow defaults).

Per-trial whale-window mean ramp rates (W/s, threshold=15000 chars, from
analyze_power_shape_whale.py on each trial's own files):
  mono (b16384):  t1=50.9  t2=53.8  t3=52.9
  chunk=512:      t1=35.2  t2=42.7  t3=43.9
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate
from ramp_reserve_procurement import collect_peak_ramps, reserve_levels

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
T_window, dt, warmup = 100.0, 0.1, 50.0

TRIALS = [
    dict(t=1, rec=f"{RAW}/2026-07-26-pesim_gate_pareto-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_pareto-b16384-t1-power.csv",
         mono_ramp=50.9, chunk_ramp=35.2),
    dict(t=2, rec=f"{RAW}/2026-07-27-pesim_gate_pareto-b16384-t2.jsonl",
         pw=f"{RAW}/2026-07-27-pesim_gate_pareto-b16384-t2-power.csv",
         mono_ramp=53.8, chunk_ramp=42.7),
    dict(t=3, rec=f"{RAW}/2026-07-27-pesim_gate_pareto-b16384-t3.jsonl",
         pw=f"{RAW}/2026-07-27-pesim_gate_pareto-b16384-t3-power.csv",
         mono_ramp=52.9, chunk_ramp=43.9),
]


def main():
    reductions = {"95%": [], "99%": [], "99.9%": []}
    single_server_reductions = []
    abs_levels = {"95%": [], "99%": [], "99.9%": []}  # (mono, chunk) MW/min per trial

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
            abs_levels[level].append((m, c))
            print(f"    {level:>6}: mono={m:.4f}  chunk=512={c:.4f} MW/min  reduction={red:.1f}%")

    print(f"\n{'='*60}\nSUMMARY ACROSS 3 TRIALS (mean +/- std)\n{'='*60}")
    ssr = np.array(single_server_reductions)
    print(f"single-server ramp reduction: {ssr.mean():.1f}% +/- {ssr.std():.1f}  "
          f"(per-trial: {[f'{x:.1f}' for x in ssr]})")
    for level, vals in reductions.items():
        v = np.array(vals)
        m_arr = np.array([a[0] for a in abs_levels[level]])
        c_arr = np.array([a[1] for a in abs_levels[level]])
        print(f"reserve reduction {level:>6}: {v.mean():.1f}% +/- {v.std():.1f}  "
              f"(per-trial: {[f'{x:.1f}' for x in v]})")
        print(f"    absolute MW/min: mono={m_arr.mean():.3f}+/-{m_arr.std():.3f}  "
              f"chunk=512={c_arr.mean():.3f}+/-{c_arr.std():.3f}")


if __name__ == "__main__":
    main()
