#!/usr/bin/env python3
"""Two-state model, but with each policy's OWN real timing (p_duty, mean_burst), instead of
sharing mono's timing for both -- the original two-state model computed `cal` once (from
mono's trace) and reused it for chunk too, an assumption directly contradicted by the data:
chunk=512's own trace gives mean_burst=3.68s / p_duty=0.696 vs mono's 1.63s / 0.561 (chunk's
whale-busy window runs >2x longer in wall-clock time, since chunking spreads the same prefill
across more decode-interleaved rounds).

Deliberately keeps the deterministic RC-filter / no-noise structure of the ORIGINAL two-state
model (not the OU continuous one) so this isolates ONE change -- per-policy timing -- rather
than re-mixing it with the separate noise/tail-behavior question already explored in
cf_continuous_regime_model.py and cf_tracebootstrap_model.py. P_b/P_max use the same
trimmed-mean calibration as those scripts (mu_b/mu_max), not the original calibrate()'s
median/max (which was shown to be contaminated by cold-start/drain transients) -- but they are
still used here as flat constants (no within-regime noise), to keep this a single-variable
change from the paper's original model.
"""
import sys

import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous
from coincidence_factor_model import simulate_states, smoothed_aggregate

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def ramp_coincidence_factor_ownstate(N, cal, tau, T_window, dt, n_mc, rng, warmup=50.0):
    """Same as coincidence_factor_model.ramp_coincidence_factor, but cal's own p_duty/
    mean_burst (not a shared one) drives simulate_states -- the only change from the
    original."""
    single_max_ramp = (cal["mu_max"] - cal["mu_b"]) / tau
    i0 = int(warmup / dt)
    peak_ramps = []
    for _ in range(n_mc):
        state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
        D = smoothed_aggregate(state, cal["mu_b"], cal["mu_max"], dt, tau)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps.append(ramp[i0:].max() if len(ramp) > i0 else 0.0)
    return np.mean(peak_ramps) / (N * single_max_ramp)


def main():
    dt = 0.1
    T_sim = 100.0
    warmup = 50.0
    rng = np.random.default_rng(20260728)

    print("=== Per-policy real timing (no shared-mono-timing assumption) ===")
    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                                     "mono (b16384)")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv",
                                      "chunk=512")

    tau_mono = (cal_mono["mu_max"] - cal_mono["mu_b"]) / 45.7
    tau_chunk = (cal_chunk["mu_max"] - cal_chunk["mu_b"]) / 30.8
    print(f"  tau_mono={tau_mono:.3f}s  tau_chunk={tau_chunk:.3f}s")

    print()
    print("=== Fleet-scale ramp-CF, N-sweep (each policy's own p_duty/mean_burst) ===")
    n_mc = 30
    for N in [1000, 5000, 10000, 20000]:
        cf_mono = ramp_coincidence_factor_ownstate(N, cal_mono, tau_mono, T_sim, dt, n_mc, rng, warmup)
        cf_chunk = ramp_coincidence_factor_ownstate(N, cal_chunk, tau_chunk, T_sim, dt, n_mc, rng, warmup)
        print(f"  N={N:<6} CF_ramp mono={cf_mono:.4f}  chunk=512={cf_chunk:.4f}  "
              f"reduction={100*(1-cf_chunk/cf_mono):.1f}%")

    sweep_utilization(cal_mono, cal_chunk, tau_mono, tau_chunk, 10000, T_sim, dt, n_mc, rng, warmup)
    sweep_partial_adoption(cal_mono, cal_chunk, tau_mono, tau_chunk, 10000, T_sim, dt, n_mc, rng, warmup)


def sweep_utilization(cal_mono, cal_chunk, tau_mono, tau_chunk, N, T_sim, dt, n_mc, rng, warmup=50.0):
    """Holds each policy's OWN measured mean_burst fixed (the "how long does one whale-busy
    window last" property, which genuinely differs by policy) but sweeps p_duty parametrically
    across a range of hypothetical fleet utilization levels, instead of using only the one
    (fairly high, ~0.56-0.70) occupancy level our lab trials happened to run at. Tests whether
    the overlap-opportunity cost that cancels chunk's smoothing benefit is itself a function of
    how saturated the fleet is -- realistic, since a real fleet spans a much wider utilization
    range than one lab configuration."""
    print()
    print("=== Utilization sensitivity: does load level change the sign? ===")
    print("(each policy keeps its OWN measured mean_burst; p_duty is swept hypothetically)")
    for p_duty in [0.1, 0.2, 0.3, 0.4, 0.5, 0.56, 0.6, 0.7]:
        cal_m = dict(cal_mono, p_duty=p_duty)
        cal_c = dict(cal_chunk, p_duty=p_duty)
        cf_mono = ramp_coincidence_factor_ownstate(N, cal_m, tau_mono, T_sim, dt, n_mc, rng, warmup)
        cf_chunk = ramp_coincidence_factor_ownstate(N, cal_c, tau_chunk, T_sim, dt, n_mc, rng, warmup)
        print(f"  p_duty={p_duty:<5} CF_ramp mono={cf_mono:.4f}  chunk={cf_chunk:.4f}  "
              f"reduction={100*(1-cf_chunk/cf_mono):.1f}%")


def mixed_fleet_ramp(N, frac_chunk, cal_mono, cal_chunk, tau_mono, tau_chunk, T_window, dt, n_mc, rng, warmup=50.0):
    """Partial-adoption fleet: frac_chunk of the N servers run chunk=512 (their own measured
    timing/tau), the rest run mono. Returns the mixed fleet's peak |dD/dt|, NOT normalized (so
    it can be compared directly against an all-mono N-server baseline -- the realistic
    question a datacenter operator actually faces: does incrementally rolling out chunk on
    SOME servers lower the fleet's total ramp versus not rolling it out at all, even if full
    rollout doesn't help)."""
    n_chunk = int(round(frac_chunk * N))
    n_mono = N - n_chunk
    i0 = int(warmup / dt)
    n_steps = int((T_window + warmup) / dt)
    peaks = []
    for _ in range(n_mc):
        D = np.zeros(n_steps)
        if n_mono > 0:
            state_m = simulate_states(n_mono, 0.0, cal_mono, T_window + warmup, dt, rng)
            D_m = smoothed_aggregate(state_m, cal_mono["mu_b"], cal_mono["mu_max"], dt, tau_mono)
            D = D + D_m
        if n_chunk > 0:
            state_c = simulate_states(n_chunk, 0.0, cal_chunk, T_window + warmup, dt, rng)
            D_c = smoothed_aggregate(state_c, cal_chunk["mu_b"], cal_chunk["mu_max"], dt, tau_chunk)
            D = D + D_c
        ramp = np.abs(np.diff(D)) / dt
        peaks.append(ramp[i0:].max() if len(ramp) > i0 else 0.0)
    return np.mean(peaks)


def sweep_partial_adoption(cal_mono, cal_chunk, tau_mono, tau_chunk, N, T_sim, dt, n_mc, rng, warmup=50.0):
    print()
    print("=== Partial adoption: does converting SOME (not all) servers to chunk help? ===")
    baseline = mixed_fleet_ramp(N, 0.0, cal_mono, cal_chunk, tau_mono, tau_chunk, T_sim, dt, n_mc, rng, warmup)
    print(f"  all-mono baseline peak ramp: {baseline:.2f} W/s")
    for frac in [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        peak = mixed_fleet_ramp(N, frac, cal_mono, cal_chunk, tau_mono, tau_chunk, T_sim, dt, n_mc, rng, warmup)
        print(f"  frac_chunk={frac:<4} peak_ramp={peak:.2f} W/s  vs all-mono: {100*(1-peak/baseline):.1f}%")


if __name__ == "__main__":
    main()
