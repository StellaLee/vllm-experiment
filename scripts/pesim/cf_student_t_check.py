#!/usr/bin/env python3
"""Quick check: does swapping Gaussian OU noise for Student's-t (heavier tails) close the
p99 gap found in the OU validation (simulated p99 ramp ~6-7x below real)? Same exact-OU
discretization, just drawing innovations from a t-distribution (rescaled to unit variance)
instead of standard normal."""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from cf_continuous_regime_model import calibrate_continuous
from coincidence_factor_model import simulate_states

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def ou_t_aggregate(state, mu_b, mu_max, sigma_b, sigma_max, dt, kappa, df, rng):
    N, n_steps = state.shape
    P = np.empty((N, n_steps), dtype=float)
    P[:, 0] = mu_b
    decay = np.exp(-kappa * dt)
    noise_scale = np.sqrt(1.0 - decay ** 2)
    t_scale = np.sqrt(df / (df - 2))  # rescale Student's-t to unit variance
    for i in range(1, n_steps):
        on = state[:, i - 1]
        mu = np.where(on, mu_max, mu_b)
        sigma = np.where(on, sigma_max, sigma_b)
        Z = rng.standard_t(df, size=N) / t_scale
        P[:, i] = mu + (P[:, i - 1] - mu) * decay + sigma * noise_scale * Z
    return P


def ramp_stats(cal, kappa, df, dt, T, rng, n_mc=20):
    means, p99s, maxes = [], [], []
    for _ in range(n_mc):
        state = simulate_states(1, 0.0, cal, T, dt, rng)
        P = ou_t_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"], cal["sigma_max"],
                            dt, kappa, df, rng)[0]
        ramp = np.abs(np.diff(P)) / dt
        means.append(ramp.mean()); p99s.append(np.percentile(ramp, 99)); maxes.append(ramp.max())
    return np.mean(means), np.mean(p99s), np.mean(maxes)


def main():
    dt, T, rng = 0.1, 100.0, np.random.default_rng(7)
    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv", "mono")
    kappa_mono = 0.1320  # already fit for Gaussian; kept fixed to isolate the effect of df alone
    print(f"target (real measured): mean=46.9  p99=856.6  max=2146.0 W/s\n")
    for df in [30, 10, 5, 3, 2.5]:
        m, p99, mx = ramp_stats(cal_mono, kappa_mono, df, dt, T, rng)
        print(f"  df={df:<5} mean={m:.1f}  p99={p99:.1f}  max={mx:.1f} W/s")


if __name__ == "__main__":
    main()
