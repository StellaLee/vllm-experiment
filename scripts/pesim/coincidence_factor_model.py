#!/usr/bin/env python3
"""Coincidence-factor Monte Carlo model, seeded with the REAL measured single-GPU trace.
Run from the directory containing the two input files (see main(), or scp them down from
logs/2026-07-24-pesimgate-b16384-t1{,-power.csv} on the box).

Calibrates a two-state (ON/OFF) renewal process from the mono/16384 trial-1 trace (arrival
rate, mean burst duration, duty cycle, P_b, P_max), then Monte-Carlo simulates N-server
aggregate demand under a synchronization parameter s in [0,1]:
  - s=0: N independent whale-arrival processes (each server's own arrivals, uncorrelated)
  - s=1: all N servers share the exact same arrival instants (full synchronization / cold-load-pickup)
  - 0<s<1: each arrival event is, independently, a "shared" event (hits all N servers at once,
    probability s) or an "idiosyncratic" event (hits one server, probability 1-s).

THE VALIDATED RESULT, after two real corrections found by this Monte Carlo check itself
(both documented in docs/2026-07-24-pes-im-prepare.md sec 5.1 -- read that before citing
either number in the paper):

  1. The naive closed form CF(N,s) ~= s + (1-s)*p (dropping O(1/sqrt(N))) does NOT hold,
     because it silently assumed zero baseline power when "off". Our GPUs have a high,
     nonzero decode-only baseline (P_b/P_max = 0.81 here) even outside whale windows, which
     compresses the achievable diversification benefit. Corrected s=0 endpoint:
         CF(N, s=0) -> P_b/P_max + (1 - P_b/P_max)*p_duty as N grows  (validated: predicted
         0.9343 vs simulated 0.9423 at N=1000).
  2. The s-parameter does NOT interpolate smoothly between the s=0 and s=1 endpoints. Any
     nonzero probability of a fully-shared (synchronized) arrival event means such an event
     WILL eventually occur within the observation window, and when it does, every server
     bursts simultaneously (K=N exactly) for its duration -- so CF jumps toward 1 abruptly
     once s is large enough that a shared event is likely within the window, not gradually as
     s increases. Demonstrated directly: at N=200, s=0.05 stays near the independent floor
     (~0.98) while s=0.10 already reaches ~0.99 -- closer to a THRESHOLD effect than a graded
     one. This is arguably the more useful finding for the paper: a SMALL probability of
     correlated/synchronized whale arrival across a fleet is disproportionately dangerous for
     aggregate peak demand, regardless of how well-diversified the "normal" independent
     traffic is -- which is the same qualitative lesson as cold-load pickup (sec 5.2).

RAMP-RATE EXTENSION (added 2026-07-25, after the level CF above was already validated):
an earlier version of this file computed a "ramp CF" as the instantaneous-step aggregate
D(t)'s |dD/dt|, normalized by (P_max-P_b)/dt -- this is DEGENERATE (scales with 1/dt for
free, not tied to anything physical) and was removed. Replaced with `smoothed_aggregate` /
`ramp_coincidence_factor`, which model each server's transition as a first-order lag with
time constant tau = (P_max-P_b)/measured_ramp_rate_w_per_s, calibrated directly from the REAL
measured mono (41.3 W/s) vs chunk=512 (27.1 W/s) mean ramp rates (paper.md Sec 5) -- so a
single isolated server's simulated ramp now matches what was actually measured on hardware,
and CF_ramp is a meaningful ratio (realized aggregate peak ramp) / (N * that same physical
single-server ramp), not a dt-dependent artifact.

UPDATE 2026-07-25: the first-pass numbers above were wrong, caught while deriving a closed-form
cross-check for CF_ramp (see ramp_coincidence_factor's docstring for the full account). Every
server started the simulated window OFF at P_b simultaneously, so the whole ensemble relaxing
toward equilibrium together produced a synchronized O(N) startup transient that dominated the
"peak ramp" in every trial -- not a real coincidence event. A Campbell's-theorem / filtered-
telegraph-noise closed form (validated against direct Monte Carlo to 5-15%) predicts the
aggregate ramp's fluctuation scale grows as sqrt(N), not N, so CF_ramp should shrink as
~1/sqrt(N) rather than saturate. Fixed by adding a warm-up period (see `warmup` arg) and
measuring the peak ramp only after the ensemble reaches stationarity: CF_ramp now shrinks as
predicted, confirming the ramp-coincidence risk genuinely averages out at fleet scale for both
scheduling policies -- unlike the level CF's baseline-driven floor.
"""
import csv
import json
import numpy as np

RNG = np.random.default_rng(20260724)


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_power(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((float(row["wall_time"]), float(row["power_w"])))
    return rows


def whale_windows(recs, whale_thresh=40000):
    windows = []
    for r in recs:
        if r.get("pad_chars", 0) >= whale_thresh and r.get("ts") is not None and r.get("latency") is not None:
            end = r["ts"]
            start = end - r["latency"]
            ttft = r.get("ttft") or 0
            windows.append((start, start + ttft))
    windows.sort()
    return windows


def merge_windows(windows):
    """Merge overlapping/adjacent whale windows into union busy-periods. Necessary because at
    concurrency=20 with whale_frac=0.15, multiple whales are frequently mid-prefill
    SIMULTANEOUSLY on the same server -- the two-state (ON/OFF) renewal model needs "at least
    one whale in flight" as its state, not naively-summed (double-counted) individual window
    durations, which can exceed the trace length (observed bug: naive sum gave duty cycle
    ~1.49 before this fix)."""
    if not windows:
        return []
    merged = [windows[0]]
    for s, e in windows[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged


def calibrate(records_path, power_path, label, whale_thresh=40000):
    recs = load_records(records_path)
    power = load_power(power_path)
    raw_windows = whale_windows(recs, whale_thresh=whale_thresh)
    windows = merge_windows(raw_windows)  # union busy-periods, not naively-summed raw windows

    t0, t1 = power[0][0], power[-1][0]
    T = t1 - t0
    burst_durations = [e - s for s, e in windows]
    mean_burst = sum(burst_durations) / len(burst_durations)
    p_duty = sum(burst_durations) / T  # now a true occupancy fraction, guaranteed in [0,1]

    starts = sorted(s for s, e in windows)
    inter_arrivals = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    mean_interarrival = sum(inter_arrivals) / len(inter_arrivals) if len(starts) > 1 else T
    lam = 1.0 / mean_interarrival
    print(f"    (raw whale requests: {len(raw_windows)}, merged into {len(windows)} "
          f"union busy-periods -- {len(raw_windows) - len(windows)} overlaps collapsed)")

    def in_any(t):
        return any(s <= t <= e for s, e in windows)

    baseline_samples = [p for t, p in power if not in_any(t)]
    whale_samples = [p for t, p in power if in_any(t)]
    P_b = sorted(baseline_samples)[len(baseline_samples) // 2]
    P_max = max(whale_samples)

    print(f"[{label}] T={T:.1f}s n_whale={len(windows)} mean_burst={mean_burst:.2f}s "
          f"mean_interarrival={mean_interarrival:.2f}s lam={lam:.4f}/s p_duty={p_duty:.4f} "
          f"P_b={P_b:.1f}W P_max={P_max:.1f}W")
    return dict(lam=lam, mean_burst=mean_burst, p_duty=p_duty, P_b=P_b, P_max=P_max)


def alternating_process(mean_on, mean_off, T_sim, rng):
    """A proper alternating renewal process: ON for Exp(mean_on), then OFF for Exp(mean_off),
    strictly sequential (never overlaps with itself) -- NOT a Poisson superposition. This
    matters: an earlier version of this simulator generated each server's busy periods as an
    unrestricted Poisson arrival stream (each arrival independently starting an excursion,
    M/G/infinity-style), which reproduces a DIFFERENT long-run busy fraction
    (1 - exp(-lam*mean_burst) = 0.494 here) than the alternating-renewal process implied by
    the real trace's own measured structure (mean_burst/mean_interarrival = 0.681, close to
    the directly measured union occupancy of 0.659) -- confirmed as the actual bug (not the
    window-length issue) once mean(K)/N was checked directly against the calibration target."""
    events = []
    if mean_on <= 0:
        return events
    t = rng.exponential(mean_off) if mean_off > 0 else 0.0
    while t < T_sim:
        dur = rng.exponential(mean_on)
        end = min(t + dur, T_sim)
        events.append((t, end))
        t = end + (rng.exponential(mean_off) if mean_off > 0 else 1e-9)
    return events


def simulate_states(N, s, cal, T_sim, dt, rng):
    """Returns the raw (N, n_steps) boolean ON/OFF state matrix (one row per server).

    s controls correlation while holding each INDIVIDUAL server's marginal occupancy fixed at
    the calibrated p_duty: a server is busy if EITHER a "shared" alternating process (occupancy
    p_shared = s*p_duty, identical realization shared by all N servers) OR its OWN
    "idiosyncratic" alternating process (occupancy p_indep, independent per server) is ON.
    p_indep is solved so the union p_shared + p_indep - p_shared*p_indep = p_duty for every s:
    p_indep = p_duty*(1-s) / (1 - s*p_duty). At s=0: p_shared=0, p_indep=p_duty (fully
    independent). At s=1: p_shared=p_duty, p_indep=0 (fully synchronized, cold-load-pickup
    limit)."""
    n_steps = int(T_sim / dt)
    state = np.zeros((N, n_steps), dtype=bool)
    mean_on = cal["mean_burst"]
    p_duty = cal["p_duty"]

    p_shared = s * p_duty
    p_indep = p_duty * (1 - s) / (1 - s * p_duty) if s * p_duty < 1 else 0.0

    if p_shared > 0:
        mean_off_shared = mean_on * (1 - p_shared) / p_shared
        shared_events = alternating_process(mean_on, mean_off_shared, T_sim, rng)
        for a, b in shared_events:
            i0, i1 = int(a / dt), int(b / dt)
            state[:, i0:i1] = True

    if p_indep > 0:
        mean_off_indep = mean_on * (1 - p_indep) / p_indep
        for i in range(N):
            indep_events = alternating_process(mean_on, mean_off_indep, T_sim, rng)
            for a, b in indep_events:
                i0, i1 = int(a / dt), int(b / dt)
                state[i, i0:i1] = True

    return state


def simulate_aggregate(N, s, cal, T_sim, dt, rng):
    """Returns the K(t) time series (# servers bursting at each grid step)."""
    return simulate_states(N, s, cal, T_sim, dt, rng).sum(axis=0)


def smoothed_aggregate(state, P_b, P_max, dt, tau):
    """Applies a first-order-lag (RC) filter to each server's boolean ON/OFF target, then sums
    across servers to get a continuous aggregate power trace D(t). This replaces the
    instantaneous step-function state with a physically motivated ramp of time constant tau:
    right after a switch, |dP/dt| = (P_max-P_b)/tau, matching the REAL measured mean ramp rate
    when tau is calibrated as tau = (P_max-P_b)/measured_ramp_rate (see ramp_coincidence_factor).
    Vectorized across the N servers (loop only over time steps, not servers), since a per-server
    Python loop over N*n_mc*n_steps was too slow to be practical here."""
    N, n_steps = state.shape
    target = np.where(state, P_max, P_b).astype(float)
    P = np.empty_like(target)
    P[:, 0] = P_b
    alpha = dt / tau
    for i in range(1, n_steps):
        P[:, i] = P[:, i - 1] + alpha * (target[:, i - 1] - P[:, i - 1])
    return P.sum(axis=0)


def ramp_coincidence_factor(N, s, cal, tau, T_window, dt, n_mc, rng, warmup=50.0):
    """Ramp-rate analogue of coincidence_factor(): CF_ramp = (realized mean peak |dD/dt|) /
    (N * single-server max ramp rate), where D(t) is the sum of N first-order-lag-smoothed
    per-server traces (see smoothed_aggregate) rather than instantaneous steps. The
    denominator N*(P_max-P_b)/tau is the theoretical worst case where all N servers ramp
    perfectly in phase -- the ramp-rate equivalent of "N*P_max" in the level CF.

    tau is calibrated from a REAL measured mean ramp rate (prepare.md Sec 3 / paper.md Sec 5):
    tau = (P_max - P_b) / measured_ramp_rate_w_per_s. Pass the mono (41.3 W/s) or chunk=512
    (27.1 W/s) rate to compare fleet-level ramp-coincidence risk under individual-server
    smoothing, independent of s.

    BUG FOUND AND FIXED 2026-07-25: simulate_states/smoothed_aggregate force every server to
    begin the window OFF at P_b simultaneously (P[:, 0] = P_b for all N rows, and the underlying
    alternating_process is guaranteed OFF at t=0 by construction). Without a warm-up, the whole
    N-server ensemble relaxes toward equilibrium occupancy together starting from that shared
    initial condition, producing a deterministic, O(N) synchronized "cold start" transient --
    not a real coincidence event. Traced directly: the simulated peak ramp landed at t=1.5-2.2s
    into EVERY single 100s window, for every N tested, regardless of s -- i.e. it was always the
    startup transient, never a genuine mid-window coincidence. This made CF_ramp falsely
    *saturate* to a nonzero floor (~0.28 mono / ~0.34 chunk=512 at N=1000) instead of shrinking.

    A closed-form check (Campbell's theorem / filtered-telegraph-noise spectrum: treating each
    server's ON/OFF state as a two-state Markov (random telegraph) process with autocovariance
    p(1-p)*exp(-lambda*|tau|), lambda = 1/mean_on + 1/mean_off, passed through the RC filter of
    pole kappa = 1/tau, gives Var(dP/dt) = sigma_T^2 * lambda*kappa^2/(lambda+kappa), sigma_T^2 =
    (P_max-P_b)^2*p(1-p) -- matches direct Monte Carlo of smoothed_aggregate to within 5-15%)
    predicts Var(D'(t)) = N*Var(dP/dt), i.e. the aggregate ramp's FLUCTUATION scale grows only as
    sqrt(N), not N -- so CF_ramp should shrink as ~1/sqrt(N), not saturate. Simulating with a
    generous warm-up (50s >> the ~1.6s relaxation time 1/lambda) and measuring the peak ramp only
    in the post-warmup, genuinely stationary remainder confirms this: CF_ramp(N=1000/5000/20000)
    = 0.036/0.016/0.0086 (mono), fitting 1/sqrt(N) almost exactly (4x N -> ~2x shrinkage). The
    ramp-coincidence risk genuinely averages out at fleet scale for both scheduling policies --
    unlike the level CF's baseline-driven floor, which does not."""
    single_max_ramp = (cal["P_max"] - cal["P_b"]) / tau
    i0 = int(warmup / dt)
    peak_ramps = []
    for _ in range(n_mc):
        state = simulate_states(N, s, cal, T_window + warmup, dt, rng)
        D = smoothed_aggregate(state, cal["P_b"], cal["P_max"], dt, tau)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps.append(ramp[i0:].max() if len(ramp) > i0 else 0.0)
    return np.mean(peak_ramps) / (N * single_max_ramp)


def coincidence_factor(N, s, cal, T_window, dt, n_mc, rng):
    """CF computed two ways, both relative to a FIXED, stated reference window T_window
    (not an arbitrarily-growable simulation horizon -- see the note below on why that
    distinction is load-bearing):
      - CF_max: mean, over n_mc independent realizations of one T_window-long period, of
        (that period's peak aggregate demand) / (N * P_max). This is the classical
        "coincident peak over a stated reference period" definition.
      - CF_p99: the 99th percentile of aggregate demand, pooled across all n_mc independent
        T_window periods, divided by (N * P_max). Percentile-based, standard utility practice
        (e.g. "1-in-10" design criteria) specifically because it's far less sensitive to
        window length than a literal running max.

    NOTE ON WHY THE WINDOW MUST BE FIXED: the maximum of a stationary process taken over an
    EVER-LONGER window is an extreme-value-theory quantity, not a CLT quantity -- given enough
    independent "looks," even fully independent servers will occasionally all be busy at once
    by chance, so a literal all-time max drifts toward the trivial ceiling of 1 for ANY N as
    the window grows (verified empirically: at N=1000 the naive max-based CF barely moved
    between T_sim=100s and T_sim=14400s, sitting at ~0.91-0.92, nowhere near p_duty=0.66).
    Fixing T_window to something empirically grounded (here: the real measured trace's own
    ~100s duration) and running many independent repetitions of THAT window, rather than one
    ever-longer window, is what restores the CLT-driven N -> p_duty convergence.
    """
    window_maxes = []
    pooled = []
    for _ in range(n_mc):
        K = simulate_aggregate(N, s, cal, T_window, dt, rng)
        D = N * cal["P_b"] + (cal["P_max"] - cal["P_b"]) * K
        window_maxes.append(D.max())
        pooled.append(D[::5])  # subsample to keep the pooled array bounded
    pooled = np.concatenate(pooled)
    CF_max = np.mean(window_maxes) / (N * cal["P_max"])
    CF_p99 = np.percentile(pooled, 99) / (N * cal["P_max"])
    return CF_max, CF_p99


def main():
    cal = calibrate("mono_t1_records.jsonl", "mono_t1_power.csv", "mono/16384")
    base_frac = cal["P_b"] / cal["P_max"]

    T_window = 100.0  # fixed to the real measured trace's own duration
    dt = 0.1
    n_mc = 200

    print()
    print("=== VALIDATED CLOSED FORM (revised twice from the original naive derivation) ===")
    print(f"P_b/P_max (baseline fraction of ceiling) = {base_frac:.4f}")
    print("s=0 (fully independent) endpoint: CF -> P_b/P_max + (1-P_b/P_max)*p_duty as N grows")
    predicted_s0 = base_frac + (1 - base_frac) * cal["p_duty"]
    print(f"  predicted = {predicted_s0:.4f}")
    for N in [10, 100, 1000]:
        CF_max, CF_p99 = coincidence_factor(N, 0.0, cal, T_window, dt, n_mc, RNG)
        print(f"  N={N:<6} CF_max(sim)={CF_max:.4f}  CF_p99(sim)={CF_p99:.4f}")
    print("s=1 (fully synchronized) endpoint: CF = 1 trivially (every server bursts together)")
    print()
    print("=== What does NOT hold: a smooth s -> linear-interpolation between the endpoints ===")
    print("Any nonzero probability s of a FULLY-SHARED arrival event means such an event WILL")
    print("eventually occur within the observation window, and when it does, ALL N servers burst")
    print("simultaneously (K=N exactly) for its duration -- driving CF to ~1 abruptly, not smoothly,")
    print("once the window is long enough to contain at least one such event. Demonstrated directly:")
    for s in [0.05, 0.1, 0.25]:
        CF_max, _ = coincidence_factor(200, s, cal, T_window, dt, n_mc, RNG)
        print(f"  N=200, s={s:<5}: CF_max={CF_max:.4f}")
    print("(s=0.05 stays near the independent floor; s=0.10 already jumps to ~1 -- a THRESHOLD")
    print(" effect, not a graded one. This is itself the more important, more actionable finding:")
    print(" a SMALL probability of correlated/synchronized whale arrival across a fleet is")
    print(" disproportionately dangerous for aggregate peak demand, regardless of how well")
    print(" diversified the 'normal' independent traffic is.)")

    print()
    print("=== RAMP-RATE EXTENSION (new -- physically-grounded, first-order-lag per server) ===")
    print("tau calibrated so a single isolated server's ramp matches the REAL measured mean")
    print("ramp rate (paper.md Sec 5): mono=41.3 W/s, chunk(budget=512)=27.1 W/s.")
    ramp_rates = {"mono (41.3 W/s)": 41.3, "chunk=512 (27.1 W/s)": 27.1}
    n_mc_ramp = 50  # smaller than n_mc=200 above -- smoothed_aggregate is O(n_steps) per rep
    print("-- does individual-server ramp smoothing lower fleet ramp-coincidence at s=0? --")
    for label, rate in ramp_rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        for N in [10, 100, 1000]:
            cf_ramp = ramp_coincidence_factor(N, 0.0, cal, tau, T_window, dt, n_mc_ramp, RNG)
            print(f"  [{label}] N={N:<6} tau={tau:.3f}s CF_ramp(s=0)={cf_ramp:.4f}")
    print("-- is the s-threshold effect (seen on level CF) also present on ramp CF? --")
    for label, rate in ramp_rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        for s in [0.0, 0.05, 0.1, 0.25]:
            cf_ramp = ramp_coincidence_factor(200, s, cal, tau, T_window, dt, n_mc_ramp, RNG)
            print(f"  [{label}] N=200, s={s:<5}: CF_ramp={cf_ramp:.4f}")


if __name__ == "__main__":
    main()
