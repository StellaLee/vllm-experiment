#!/usr/bin/env python3
"""Day-3 timeboxed prototype: dynamic per-burst ramp-cap controller (Option B from the PES-IM
controller discussion), evaluated purely in simulation (no new GPU experiments) by extending
coincidence_factor_model.py's already-calibrated N-server Monte Carlo model.

CONTROL STRATEGY: each server follows the same calibrated ON/OFF arrival timing as the existing
model (simulate_states, unchanged -- arrival timing is a workload property, not a policy choice).
The only new decision: at the instant any server STARTS a new whale burst, the controller checks
a causal (backward-looking) EWMA estimate of the facility's current aggregate ramp rate against a
cap C. Under cap -> that burst gets tau_mono (fast/cheap); at/over cap -> tau_chunk (smooth). The
decision is made once per burst (persists for that burst's duration), for only the server(s)
actually starting a transition at that moment.

Baselines: static-all-mono, static-all-chunk, dynamic controller. Metrics: P(cap violation) (peak
ramp exceeds C), "cost" (fraction of bursts forced into chunk), P99/max realized ramp. Records
chunk-fraction-over-time so the controller's behavior can be sanity-checked / plotted like the
paper's existing power timelines.

Go/no-go: the controller must beat BOTH static baselines simultaneously on (violation-rate, cost)
-- not merely land between them, which any interpolation would do trivially.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate, simulate_states

RNG = np.random.default_rng(20260807)


def simulate_run(N, s, cal, tau_mono, tau_chunk, dt, T_window, warmup, rng,
                  mode="static_mono", cap=None, ewma_tau=2.0):
    """mode in {"static_mono", "static_chunk", "dynamic"}. Returns dict with D, ramp, chunk_frac_t
    (all post-warmup) and chunk_fraction (fraction of BURSTS assigned chunk over the whole run)."""
    T_sim = T_window + warmup
    n_steps = int(T_sim / dt)
    state = simulate_states(N, s, cal, T_sim, dt, rng)  # (N, n_steps) bool
    P_b, P_max = cal["P_b"], cal["P_max"]
    target = np.where(state, P_max, P_b).astype(float)

    P = np.empty((N, n_steps))
    P[:, 0] = P_b
    tau_active = np.full(N, tau_mono if mode != "static_chunk" else tau_chunk)
    prev_state = np.zeros(N, dtype=bool)  # guaranteed OFF at t=0 by construction

    ewma_alpha = dt / ewma_tau
    ramp_ewma = 0.0
    n_chunk_bursts = 0
    n_total_bursts = 0
    D = np.empty(n_steps)
    D[0] = P[:, 0].sum()
    chunk_frac_t = np.zeros(n_steps)
    chunk_frac_t[0] = np.mean(tau_active == tau_chunk)

    for t in range(1, n_steps):
        onset = state[:, t] & ~prev_state
        if onset.any() and mode == "dynamic":
            decide_chunk = ramp_ewma > cap
            tau_active[onset] = tau_chunk if decide_chunk else tau_mono
            k = int(onset.sum())
            n_total_bursts += k
            if decide_chunk:
                n_chunk_bursts += k
        elif onset.any():
            n_total_bursts += int(onset.sum())
            if mode == "static_chunk":
                n_chunk_bursts += int(onset.sum())

        alpha = dt / tau_active
        P[:, t] = P[:, t - 1] + alpha * (target[:, t - 1] - P[:, t - 1])
        D[t] = P[:, t].sum()

        inst_ramp = abs(D[t] - D[t - 1]) / dt
        ramp_ewma = ewma_alpha * inst_ramp + (1 - ewma_alpha) * ramp_ewma
        chunk_frac_t[t] = np.mean(tau_active == tau_chunk)
        prev_state = state[:, t]

    i0 = int(warmup / dt)
    ramp = np.abs(np.diff(D)) / dt
    chunk_fraction = n_chunk_bursts / n_total_bursts if n_total_bursts else 0.0
    return dict(
        D=D[i0:], ramp=ramp[max(i0 - 1, 0):], chunk_frac_t=chunk_frac_t[i0:],
        chunk_fraction=chunk_fraction, peak_ramp=ramp[max(i0 - 1, 0):].max(),
    )


def main():
    cal = calibrate(
        "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/mono_t1_records.jsonl",
        "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/mono_t1_power.csv",
        "mono/16384",
    )
    N = 1000
    s = 0.0
    T_window, dt, warmup = 100.0, 0.1, 50.0
    n_mc = 30
    tau_mono = (cal["P_max"] - cal["P_b"]) / 41.3
    tau_chunk = (cal["P_max"] - cal["P_b"]) / 27.1

    print(f"N={N} tau_mono={tau_mono:.3f}s tau_chunk={tau_chunk:.3f}s")

    # Step 1: characterize the two static baselines' peak-ramp distributions.
    mono_peaks, chunk_peaks = [], []
    for _ in range(n_mc):
        mono_peaks.append(simulate_run(N, s, cal, tau_mono, tau_chunk, dt, T_window, warmup, RNG,
                                        mode="static_mono")["peak_ramp"])
        chunk_peaks.append(simulate_run(N, s, cal, tau_mono, tau_chunk, dt, T_window, warmup, RNG,
                                         mode="static_chunk")["peak_ramp"])
    mono_peaks, chunk_peaks = np.array(mono_peaks), np.array(chunk_peaks)
    print(f"static-mono  peak ramp: mean={mono_peaks.mean():.1f} W/s  "
          f"p50={np.median(mono_peaks):.1f}  p90={np.percentile(mono_peaks,90):.1f}")
    print(f"static-chunk peak ramp: mean={chunk_peaks.mean():.1f} W/s  "
          f"p50={np.median(chunk_peaks):.1f}  p90={np.percentile(chunk_peaks,90):.1f}")

    # Step 2: set a cap that's meaningfully between the two -- e.g. static-chunk's own P90 peak
    # ramp, so static-chunk mostly satisfies it (by construction, ~90% of the time) and static-mono
    # mostly violates it (since chunk_p90 << mono's typical peak).
    cap = np.percentile(chunk_peaks, 90)
    print(f"\ncap C = static-chunk's P90 peak ramp = {cap:.1f} W/s")
    mono_violation = np.mean(mono_peaks > cap)
    chunk_violation = np.mean(chunk_peaks > cap)
    print(f"static-mono  violation rate: {mono_violation:.2f}  (cost: 0.00, always mono)")
    print(f"static-chunk violation rate: {chunk_violation:.2f}  (cost: 1.00, always chunk)")

    # Step 3: run the dynamic controller under the same cap.
    dyn_peaks, dyn_costs = [], []
    for _ in range(n_mc):
        r = simulate_run(N, s, cal, tau_mono, tau_chunk, dt, T_window, warmup, RNG,
                          mode="dynamic", cap=cap)
        dyn_peaks.append(r["peak_ramp"])
        dyn_costs.append(r["chunk_fraction"])
    dyn_peaks, dyn_costs = np.array(dyn_peaks), np.array(dyn_costs)
    dyn_violation = np.mean(dyn_peaks > cap)
    print(f"\ndynamic ctrl violation rate: {dyn_violation:.2f}  "
          f"(cost: {dyn_costs.mean():.2f} mean fraction of bursts forced to chunk)")
    print(f"dynamic ctrl peak ramp: mean={dyn_peaks.mean():.1f} W/s  "
          f"p50={np.median(dyn_peaks):.1f}  p90={np.percentile(dyn_peaks,90):.1f}")

    print("\n=== GO/NO-GO ===")
    beats_mono_violation = dyn_violation < mono_violation
    beats_chunk_cost = dyn_costs.mean() < 1.0
    print(f"beats static-mono on violation rate ({dyn_violation:.2f} < {mono_violation:.2f})? "
          f"{beats_mono_violation}")
    print(f"beats static-chunk on cost ({dyn_costs.mean():.2f} < 1.00)? {beats_chunk_cost}")
    print(f"PURSUE FURTHER: {beats_mono_violation and beats_chunk_cost}")

    # Save one example run's time series (chunk fraction over time) for a sanity-check plot.
    example = simulate_run(N, s, cal, tau_mono, tau_chunk, dt, T_window, warmup,
                            np.random.default_rng(1), mode="dynamic", cap=cap)
    np.savez("/tmp/ramp_cap_controller_example.npz",
             D=example["D"], ramp=example["ramp"], chunk_frac_t=example["chunk_frac_t"])
    print("\nsaved example dynamic-run time series (D, ramp, chunk_frac_t) to "
          "/tmp/ramp_cap_controller_example.npz")


if __name__ == "__main__":
    main()
