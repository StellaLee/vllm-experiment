#!/usr/bin/env python3
"""Quick check: OU diffusion + a compound-Poisson jump term (a discrete, rare, large event
superimposed on the continuous mean-reverting process) -- testing whether this closes the
tail gap that Student's-t (fatter but still continuous noise) could not."""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous
from coincidence_factor_model import simulate_states

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def ou_jump_aggregate(state, mu_b, mu_max, sigma_b, sigma_max, dt, kappa, jump_rate, jump_scale, rng):
    """jump_rate: expected jumps per second (Poisson). jump_scale: mean |jump size| in W,
    drawn from a zero-mean Laplace (symmetric, so it can push ramp up or down) -- deliberately
    simple (2 parameters) rather than fitting a full distribution shape."""
    N, n_steps = state.shape
    P = np.empty((N, n_steps), dtype=float)
    P[:, 0] = mu_b
    decay = np.exp(-kappa * dt)
    noise_scale = np.sqrt(1.0 - decay ** 2)
    p_jump = jump_rate * dt
    for i in range(1, n_steps):
        on = state[:, i - 1]
        mu = np.where(on, mu_max, mu_b)
        sigma = np.where(on, sigma_max, sigma_b)
        Z = rng.standard_normal(N)
        base = mu + (P[:, i - 1] - mu) * decay + sigma * noise_scale * Z
        jump_mask = rng.random(N) < p_jump
        jump_size = rng.laplace(0, jump_scale, size=N)
        P[:, i] = base + jump_mask * jump_size
    return P


def ramp_stats(cal, kappa, jump_rate, jump_scale, dt, T, rng, n_mc=20):
    means, p99s, maxes = [], [], []
    for _ in range(n_mc):
        state = simulate_states(1, 0.0, cal, T, dt, rng)
        P = ou_jump_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"], cal["sigma_max"],
                               dt, kappa, jump_rate, jump_scale, rng)[0]
        ramp = np.abs(np.diff(P)) / dt
        means.append(ramp.mean()); p99s.append(np.percentile(ramp, 99)); maxes.append(ramp.max())
    return np.mean(means), np.mean(p99s), np.mean(maxes)


def main():
    dt, T, rng = 0.1, 100.0, np.random.default_rng(7)
    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv", "mono")
    kappa_mono = 0.1320
    print(f"target (real measured): mean=46.9  p99=856.6  max=2146.0 W/s\n")
    for jump_rate, jump_scale in [(0.05, 30), (0.1, 30), (0.1, 50), (0.2, 40), (0.3, 40)]:
        m, p99, mx = ramp_stats(cal_mono, kappa_mono, jump_rate, jump_scale, dt, T, rng)
        print(f"  jump_rate={jump_rate}/s  jump_scale={jump_scale}W  ->  mean={m:.1f}  p99={p99:.1f}  max={mx:.1f} W/s")


if __name__ == "__main__":
    main()
