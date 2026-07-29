#!/usr/bin/env python3
"""Robustness check: does the fleet-scale coincidence-factor / reserve-procurement finding
survive recalibrating the Monte Carlo model from two DIFFERENT single-GPU traces already
collected for the paper's Section 3.1 robustness checks (docs/2026-07-24-pes-im-prepare.md),
instead of the original narrow-variance closed-loop trace?

  (a) widened request-size distribution (short cv2 0.5->3.0; whale range [44000,50000]->
      [18000,50000], cv 0.037->0.26) but SAME closed-loop concurrency=20 arrivals
  (b) same widened distribution, but Poisson open-loop arrivals (--rate 2.4, no admission cap)

No new GPU experiments: both traces already exist in data/pesim_gate_raw/ (single trial each,
2026-07-25). whale_thresh is lowered from the model's default 40000 to 15000 for these two
conditions -- the widened whale floor is 18000 chars and the widened short-prompt ceiling is
12000 chars, so 15000 sits cleanly in the gap (verified: non-whale region's frac_above(420W)
drops to 0.0000 at threshold=15000 for both conditions, matching the paper's own robustness
writeup's description of near-zero non-whale duty cycle -- confirming 15000 is the right cut,
not 40000 which mixes some genuine whale prefill into the "non-whale/baseline" bucket).

Mono/chunk=512 ramp rates (W/s, whale-window-disaggregated, threshold=15000, matches this
script's own calibration threshold) were computed by:
  DATE=2026-07-25 NAME=pesim_gate_wide[_poisson] LOGDIR=data/pesim_gate_raw \\
  BUDGETS="16384 512" TRIALS="1" WHALE_THRESH_CHARS=15000 python3 scripts/pesim/analyze_power_shape_whale.py
  widened closed-loop: mono=50.4 W/s, chunk=512=29.1 W/s
  widened Poisson:     mono=50.3 W/s, chunk=512=19.1 W/s
"""
import sys
import time
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from coincidence_factor_model import calibrate, ramp_coincidence_factor
from ramp_reserve_procurement import collect_peak_ramps, reserve_levels

RNG = np.random.default_rng(20260807)
RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"

CONDITIONS = [
    dict(label="widened closed-loop (conc=20)",
         rec=f"{RAW}/2026-07-25-pesim_gate_wide-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-25-pesim_gate_wide-b16384-t1-power.csv",
         mono_ramp=50.4, chunk_ramp=29.1),
    dict(label="widened Poisson open-loop",
         rec=f"{RAW}/2026-07-25-pesim_gate_wide_poisson-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-25-pesim_gate_wide_poisson-b16384-t1-power.csv",
         mono_ramp=50.3, chunk_ramp=19.1),
]

T_window, dt, warmup = 100.0, 0.1, 50.0


def main():
    for cond in CONDITIONS:
        label = cond["label"]
        print(f"\n{'='*70}\n{label}\n{'='*70}")
        cal = calibrate(cond["rec"], cond["pw"], label, whale_thresh=15000)
        tau_mono = (cal["P_max"] - cal["P_b"]) / cond["mono_ramp"]
        tau_chunk = (cal["P_max"] - cal["P_b"]) / cond["chunk_ramp"]
        print(f"  tau_mono={tau_mono:.3f}s  tau_chunk={tau_chunk:.3f}s  "
              f"(single-server ramp reduction: {(1 - cond['chunk_ramp']/cond['mono_ramp'])*100:.1f}%)")

        print("\n  -- CF_ramp(N) at s=0: does it shrink as N grows for BOTH policies? --")
        n_mc_ramp = 50
        for N in [10, 100, 1000, 5000]:
            cf_mono = ramp_coincidence_factor(N, 0.0, cal, tau_mono, T_window, dt, n_mc_ramp, RNG, warmup)
            cf_chunk = ramp_coincidence_factor(N, 0.0, cal, tau_chunk, T_window, dt, n_mc_ramp, RNG, warmup)
            print(f"    N={N:<6} CF_ramp mono={cf_mono:.4f}  chunk=512={cf_chunk:.4f}")

        print("\n  -- Ramping-reserve procurement at N=10,000 (300 MC trials) --")
        N, n_mc = 10000, 300
        t0 = time.time()
        mono_peaks = collect_peak_ramps(N, 0.0, cal, tau_mono, T_window, dt, n_mc, RNG, warmup)
        chunk_peaks = collect_peak_ramps(N, 0.0, cal, tau_chunk, T_window, dt, n_mc, RNG, warmup)
        print(f"  (simulated in {time.time()-t0:.1f}s)")
        mono_r = reserve_levels(mono_peaks, f"{label}: mono")
        chunk_r = reserve_levels(chunk_peaks, f"{label}: chunk=512")
        print(f"\n  === reserve comparison, {label} ===")
        for level, key in [("95%", "R95"), ("99%", "R99"), ("99.9%", "R999")]:
            m, c = mono_r[key], chunk_r[key]
            reduction = (1 - c / m) * 100
            print(f"    {level:>6} reliability: mono={m:.4f}  chunk=512={c:.4f} MW/min  "
                  f"reduction={reduction:.1f}%")


if __name__ == "__main__":
    main()
