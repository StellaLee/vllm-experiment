#!/usr/bin/env python3
"""Chunk=2048 reserve-procurement number under the corrected model (per-policy timing + OU
noise), same 3-real-trial methodology as cf_reserve_final_3trial.py, for the paper's budget-
choice caveat sentence (currently cites an uncorrected ~7% figure)."""
import sys
import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous, ou_continuous_aggregate
from coincidence_factor_model import simulate_states
from cf_reserve_final_3trial import fit_kappa, collect_peak_ramps, reserve_levels, MONO_TRIALS

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
RNG = np.random.default_rng(20260807)

TRIALS_2048 = [
    ("2026-07-26-pesim_gate_wide-b2048-t1", 45.06),
    ("2026-07-26-pesim_gate_wide-b2048-t2", 45.06),
    ("2026-07-26-pesim_gate_wide-b2048-t3", 45.06),
]


def main():
    N = 10000
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc_fit, n_mc_reserve = 25, 120

    mono_reserves, c2048_reserves = [], []
    for (tag, target), lst in [(m, mono_reserves) for m in MONO_TRIALS] + \
                               [(c, c2048_reserves) for c in TRIALS_2048]:
        cal = calibrate_continuous(f"{RAW}/{tag}.jsonl", f"{RAW}/{tag}-power.csv", tag, whale_thresh=15000)
        kappa = fit_kappa(target, cal, dt, T_window, warmup, n_mc_fit, RNG)
        peaks = collect_peak_ramps(N, cal, kappa, T_window, dt, n_mc_reserve, RNG, warmup)
        r95, r99, r999 = reserve_levels(peaks)
        print(f"[{tag}] kappa={kappa:.4f} R95={r95:.4f} R99={r99:.4f} R999={r999:.4f} MW/min")
        lst.append((r95, r99, r999))

    mono_arr, c2048_arr = np.array(mono_reserves), np.array(c2048_reserves)
    print("\n=== chunk=2048 vs mono, per-policy timing + OU noise ===")
    for i, level in enumerate(["95%", "99%", "99.9%"]):
        reductions = 1 - c2048_arr[:, i] / mono_arr[:, i]
        print(f"  {level:>6}: reduction={reductions.mean()*100:.1f}%+/-{reductions.std()*100:.1f}")


if __name__ == "__main__":
    main()
