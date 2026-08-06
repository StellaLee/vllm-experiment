#!/usr/bin/env python3
"""Reserve-procurement check for the CONC=10 "mild saturation" condition -- the 3rd data point
for Appendix F's load-sensitivity analysis (CONC=20 saturated / CONC=10 mild / CONC=6
sub-saturated). Same corrected per-policy-timing + OU-noise methodology as
cf_reserve_final_3trial.py, using the self-consistent kappa/warmup fit (cf_continuous_regime_
model.fit_kappa_self_consistent) since the fixed-50s-warmup version was shown to bias results.

Whole-trace mean ramp targets (W/s, dt=0.1s grid, 5s trim each side):
  mono (b16384):  t1=43.30  t2=45.00  t3=42.39
  chunk=512:      t1=30.15  t2=38.95  t3=35.58
"""
import sys

import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from cf_continuous_regime_model import calibrate_continuous, fit_kappa_self_consistent, ou_continuous_aggregate
from coincidence_factor_model import simulate_states

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
RNG = np.random.default_rng(20260807)

MONO_TRIALS = [
    ("2026-07-30-pesim_gate_wide_mildsat-b16384-t1", 43.30),
    ("2026-07-30-pesim_gate_wide_mildsat-b16384-t2", 45.00),
    ("2026-07-30-pesim_gate_wide_mildsat-b16384-t3", 42.39),
]
CHUNK_TRIALS = [
    ("2026-07-30-pesim_gate_wide_mildsat-b512-t1", 30.15),
    ("2026-07-30-pesim_gate_wide_mildsat-b512-t2", 38.95),
    ("2026-07-30-pesim_gate_wide_mildsat-b512-t3", 35.58),
]
WHALE_THRESH = 15000


def collect_peak_ramps(N, cal, kappa, T_window, dt, n_mc, rng, warmup):
    i0 = int(warmup / dt)
    peaks = np.empty(n_mc)
    for k in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                     cal["sigma_max"], dt, kappa, rng)
        D = P.sum(axis=0)
        ramp = np.abs(np.diff(D)) / dt
        peaks[k] = ramp[i0:].max() if len(ramp) > i0 else 0.0
    return peaks


def reserve_levels(peak_ramps_w_per_s):
    mw_per_min = peak_ramps_w_per_s * 60.0 / 1e6
    emp_95, emp_99 = np.percentile(mw_per_min, 95), np.percentile(mw_per_min, 99)
    loc, scale = gumbel_r.fit(mw_per_min)
    fit_999 = gumbel_r.ppf(0.999, loc=loc, scale=scale)
    return emp_95, emp_99, fit_999


def main():
    N = 10000
    T_window, dt = 100.0, 0.1
    n_mc_fit = 25
    n_mc_reserve = 120

    mono_reserves, chunk_reserves = [], []
    for (tag, target), lst, kind in [(m, mono_reserves, "mono") for m in MONO_TRIALS] + \
                                     [(c, chunk_reserves, "chunk") for c in CHUNK_TRIALS]:
        cal = calibrate_continuous(f"{RAW}/{tag}.jsonl", f"{RAW}/{tag}-power.csv", tag,
                                    whale_thresh=WHALE_THRESH)
        kappa, r, warmup = fit_kappa_self_consistent(target, cal, dt, T_window, n_mc_fit, RNG)
        peaks = collect_peak_ramps(N, cal, kappa, T_window, dt, n_mc_reserve, RNG, warmup)
        r95, r99, r999 = reserve_levels(peaks)
        print(f"[{kind} {tag}] kappa={kappa:.4f} warmup={warmup:.1f}s "
              f"R95={r95:.4f} R99={r99:.4f} R999={r999:.4f} MW/min")
        lst.append((r95, r99, r999))

    mono_arr = np.array(mono_reserves)
    chunk_arr = np.array(chunk_reserves)
    print()
    print("=== CONC=10 mild-saturation: per-policy timing + OU noise, mean +/- std across 3 trials ===")
    for i, level in enumerate(["95%", "99%", "99.9%"]):
        m_mean, m_std = mono_arr[:, i].mean(), mono_arr[:, i].std()
        c_mean, c_std = chunk_arr[:, i].mean(), chunk_arr[:, i].std()
        reductions = 1 - chunk_arr[:, i] / mono_arr[:, i]
        red_mean, red_std = reductions.mean() * 100, reductions.std() * 100
        print(f"  {level:>6}: mono={m_mean:.3f}+/-{m_std:.3f}  chunk={c_mean:.3f}+/-{c_std:.3f}  "
              f"reduction={red_mean:.1f}%+/-{red_std:.1f}")


if __name__ == "__main__":
    main()
