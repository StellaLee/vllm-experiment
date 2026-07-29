#!/usr/bin/env python3
"""Final reserve-procurement numbers for the corrected model (per-policy timing + OU noise),
computed the SAME way as the original table: separately calibrate mono and chunk from EACH of
their 3 real trials, run the reserve pipeline per trial-pair, report mean +/- std across the 3 --
not just Monte Carlo noise on one fixed calibration."""
import sys
import numpy as np
from scipy.stats import gumbel_r

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous, ou_continuous_aggregate
from coincidence_factor_model import simulate_states

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
RNG = np.random.default_rng(20260807)

MONO_TRIALS = [
    ("2026-07-26-pesim_gate_wide-b16384-t1", 45.7),
    ("2026-07-26-pesim_gate_wide-b16384-t2", 45.7),
    ("2026-07-26-pesim_gate_wide-b16384-t3", 45.7),
]
CHUNK_TRIALS = [
    ("2026-07-26-pesim_gate_wide-b512-t1", 30.8),
    ("2026-07-26-pesim_gate_wide-b512-t2", 30.8),
    ("2026-07-26-pesim_gate_wide-b512-t3", 30.8),
]


def fit_kappa(target_ramp, cal, dt, T_sim, warmup, n_mc, rng, lo=0.01, hi=2.0, tol=0.02, max_iter=20):
    def mean_ramp(kappa):
        i0 = int(warmup / dt)
        ramps = []
        for _ in range(n_mc):
            state = simulate_states(1, 0.0, cal, T_sim + warmup, dt, rng)
            P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                         cal["sigma_max"], dt, kappa, rng)[0]
            ramps.append((np.abs(np.diff(P[i0:])) / dt).mean())
        return np.mean(ramps)
    r_lo, r_hi = mean_ramp(lo), mean_ramp(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        r_mid = mean_ramp(mid)
        if abs(r_mid - target_ramp) / target_ramp < tol:
            return mid
        if r_mid < target_ramp:
            lo = mid
        else:
            hi = mid
    return mid


def collect_peak_ramps(N, cal, kappa, T_window, dt, n_mc, rng, warmup=50.0):
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
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc_fit = 25
    n_mc_reserve = 120

    mono_reserves = []  # list of (R95, R99, R999)
    chunk_reserves = []

    for (tag, target), lst, kind in [(m, mono_reserves, "mono") for m in MONO_TRIALS] + \
                                     [(c, chunk_reserves, "chunk") for c in CHUNK_TRIALS]:
        cal = calibrate_continuous(f"{RAW}/{tag}.jsonl", f"{RAW}/{tag}-power.csv", tag,
                                    whale_thresh=15000)
        kappa = fit_kappa(target, cal, dt, T_window, warmup, n_mc_fit, RNG)
        peaks = collect_peak_ramps(N, cal, kappa, T_window, dt, n_mc_reserve, RNG, warmup)
        r95, r99, r999 = reserve_levels(peaks)
        print(f"[{kind} {tag}] kappa={kappa:.4f} R95={r95:.4f} R99={r99:.4f} R999={r999:.4f} MW/min")
        lst.append((r95, r99, r999))

    mono_arr = np.array(mono_reserves)   # (3, 3)
    chunk_arr = np.array(chunk_reserves)
    print()
    print("=== FINAL: per-policy timing + OU noise, mean +/- std across 3 real trials each ===")
    for i, level in enumerate(["95%", "99%", "99.9%"]):
        m_mean, m_std = mono_arr[:, i].mean(), mono_arr[:, i].std()
        c_mean, c_std = chunk_arr[:, i].mean(), chunk_arr[:, i].std()
        reductions = 1 - chunk_arr[:, i] / mono_arr[:, i]  # per-trial-pair reduction (paired by index)
        red_mean, red_std = reductions.mean() * 100, reductions.std() * 100
        print(f"  {level:>6}: mono={m_mean:.3f}+/-{m_std:.3f}  chunk={c_mean:.3f}+/-{c_std:.3f}  "
              f"reduction={red_mean:.1f}%+/-{red_std:.1f}")


if __name__ == "__main__":
    main()
