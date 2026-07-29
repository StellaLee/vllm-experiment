#!/usr/bin/env python3
"""Trace-driven bootstrap fleet-scale model -- Option A, built after the parametric continuous
(OU-noise) model raised two open questions it couldn't resolve on its own: (1) real single-server
ramp has much heavier tails than Gaussian OU noise reproduces (simulated p99 ramp ~147 W/s vs
real ~857 W/s for mono), and (2) the OU noise was assumed i.i.d. across servers with no
justification, in a paper whose OWN headline finding (the s-threshold effect on level CF) says
independence assumptions in exactly this kind of aggregation are dangerous to trust blindly.

Fix: stop parametrizing server power at all. Directly resample REAL measured continuous
power traces (all real ~80-84s trials of the SAME "wide" closed-loop condition -- 4 real mono
trials, 4 real chunk=512 trials) to build each of the N virtual servers' power signal, phase-
randomized (independent random circular shift per virtual server) so servers aren't literally
synchronized to real wall-clock time. This preserves 100% of the real trace's dynamics --
actual tail behavior, actual autocorrelation, whatever true (non-Gaussian, possibly
non-stationary) structure exists -- with no distributional assumption at all. Correlation
across the fleet (an s-like knob) is implemented by controlling what fraction of virtual
servers share the SAME (trace, phase) draw vs. get independent draws.

Caveat stated plainly: the "population" being resampled from is only 4 real trials per arm
(~80s each) -- a small library. This bootstrap is exact for the trace variability actually
observed in those 4 trials; it does NOT invent variability beyond what was measured, so it
likely UNDERESTIMATES real fleet diversity (a production fleet running for hours would see
more distinct conditions than 4 90-second lab windows capture). That is a sampling-size
limitation to flag, not a modeling-choice limitation like the previous two models had.
"""
import csv
import sys

import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"

#  NOTE: the 2026-07-25 and 2026-07-26 "t1" files are byte-identical (confirmed via md5) --
#  the 07-26 rerun evidently reused the same t1 result rather than generating a fresh one.
#  Counting both would silently double-weight that one trial in the bootstrap library, so
#  only the distinct trials are listed. Library grown 2026-07-28 in two batches: first 17 more
#  real trials per arm (t4-t20) to check whether the earlier n=3 result was small-sample noise,
#  then 30 more (t21-t50) to keep testing convergence -- 50 distinct real trials per arm total.
MONO_TRACES = [
    f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
    f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2-power.csv",
    f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3-power.csv",
] + [f"{RAW}/2026-07-28-pesim_gate_wide-b16384-t{i}-power.csv" for i in range(4, 51)]
CHUNK_TRACES = [
    f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv",
    f"{RAW}/2026-07-26-pesim_gate_wide-b512-t2-power.csv",
    f"{RAW}/2026-07-26-pesim_gate_wide-b512-t3-power.csv",
] + [f"{RAW}/2026-07-28-pesim_gate_wide-b512-t{i}-power.csv" for i in range(4, 51)]


def load_trimmed_resampled(path, dt, steady_thresh=300.0):
    """Load a real power trace and resample onto a uniform dt grid via linear interpolation
    (real NVML samples land at ~55ms median but jittery intervals, not exactly on a grid).

    Trims cold-start/drain-down transients ADAPTIVELY (first/last index crossing
    steady_thresh W), not by a fixed time window -- a fixed 5s trim was too short for some
    trials (t2/t3 of this same condition have an 8-10s cold start, confirmed by inspecting
    their per-5s power breakdown directly: still <90W at the 5-10s mark)."""
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((float(row["wall_time"]), float(row["power_w"])))
    t = np.array([r[0] for r in rows])
    p = np.array([r[1] for r in rows])
    above = np.where(p >= steady_thresh)[0]
    i0, i1 = above[0], above[-1]
    t0, t1 = t[i0], t[i1]
    grid = np.arange(t0, t1, dt)
    return np.interp(grid, t, p)


def build_library(paths, dt):
    """Returns (n_traces, min_len) WITHOUT tiling/wraparound -- sample_fleet only draws
    windows that fit entirely inside a trace (see its docstring for why: circularly tiling a
    ~70s trace to cover a 100s window created a fake seam, e.g. one trace's tail (439W) meets
    its own head (71W) -- a 368W fabricated "ramp" every wraparound, not a real event)."""
    traces = [load_trimmed_resampled(p, dt) for p in paths]
    min_len = min(len(t) for t in traces)
    return np.stack([t[:min_len] for t in traces])  # (n_traces, min_len)


def sample_fleet(library, N, n_steps, s, rng):
    """Build the (N, n_steps) aggregate-ready matrix by giving each virtual server a
    real trace (row of `library`) starting at a random phase, WITHOUT wraparound -- phase is
    drawn only from [0, lib_len-n_steps] so every virtual server's window sits entirely inside
    one real trace, never crossing the (fake) seam a circular tiling would create. This means
    n_steps must be <= lib_len (T_sim capped by the shortest available real trial after
    adaptive trimming, ~60-65s here -- shorter than the 100s window used elsewhere in the
    paper; a real constraint of only having 4 real ~70-84s trials to draw from, not a choice).

    s controls correlation: a fraction s of servers all share the SAME (trace, phase) draw
    (mirrors the existing s parameter's shared/idiosyncratic split in the parametric models);
    the rest draw independently."""
    n_traces, lib_len = library.shape
    assert n_steps <= lib_len, f"n_steps={n_steps} exceeds shortest trace length={lib_len}"
    max_phase = lib_len - n_steps

    n_shared = int(round(s * N))
    n_indep = N - n_shared
    out = np.empty((N, n_steps))

    if n_shared > 0:
        shared_trace = rng.integers(0, n_traces)
        shared_phase = rng.integers(0, max_phase + 1)
        out[:n_shared] = library[shared_trace, shared_phase:shared_phase + n_steps]

    if n_indep > 0:
        trace_ids = rng.integers(0, n_traces, size=n_indep)
        phases = rng.integers(0, max_phase + 1, size=n_indep)
        # Vectorized gather (per-server loop over N=20000 was the bottleneck): idx[i,j] =
        # phases[i]+j indexes into library's time axis, trace_ids broadcasts over the row axis.
        idx = phases[:, None] + np.arange(n_steps)[None, :]
        out[n_shared:] = library[trace_ids[:, None], idx]

    return out


def ramp_cf_bootstrap(library, N, s, dt, n_steps, target_ramp, n_mc, rng):
    peak_ramps = []
    for _ in range(n_mc):
        P = sample_fleet(library, N, n_steps, s, rng)
        D = P.sum(axis=0)
        ramp = np.abs(np.diff(D)) / dt
        peak_ramps.append(ramp.max())
    return np.mean(peak_ramps) / (N * target_ramp)


def single_server_ramp_stats(library, dt, n_mc, n_steps, rng):
    n_traces, lib_len = library.shape
    max_phase = lib_len - n_steps
    means, maxes, p99s = [], [], []
    for _ in range(n_mc):
        tid = rng.integers(0, n_traces)
        ph = rng.integers(0, max_phase + 1)
        p = library[tid, ph:ph + n_steps]
        ramp = np.abs(np.diff(p)) / dt
        means.append(ramp.mean()); maxes.append(ramp.max()); p99s.append(np.percentile(ramp, 99))
    return np.mean(means), np.mean(maxes), np.mean(p99s)


def main():
    dt = 0.1
    rng = np.random.default_rng(20260728)

    print("=== Step 1: build real-trace libraries (adaptively trimmed, resampled to dt=0.1s grid) ===")
    lib_mono = build_library(MONO_TRACES, dt)
    lib_chunk = build_library(CHUNK_TRACES, dt)
    print(f"  mono:      {lib_mono.shape[0]} real trials, {lib_mono.shape[1]} samples each "
          f"({lib_mono.shape[1]*dt:.0f}s usable)")
    print(f"  chunk=512: {lib_chunk.shape[0]} real trials, {lib_chunk.shape[1]} samples each "
          f"({lib_chunk.shape[1]*dt:.0f}s usable)")

    # T_sim capped below the shortest trace (no-wraparound sampling needs n_steps <= lib_len,
    # with room left over for phase diversity -- half the shortest usable length is a
    # reasonable balance) -- shorter than the 100s window used elsewhere in the paper, a real
    # constraint of only having 4 real ~70-84s trials, not a free choice.
    shortest = min(lib_mono.shape[1], lib_chunk.shape[1]) * dt
    T_sim = round(shortest * 0.5, -1)  # round to nearest 10s
    n_steps = int(T_sim / dt)
    print(f"  T_sim={T_sim:.0f}s (capped at half the shortest usable trace, {shortest:.0f}s, "
          f"to leave phase-diversity room)")

    print()
    print("=== Step 2: single-server ramp stats -- sanity check vs. real measured values ===")
    n_mc_check = 40
    m_mean, m_max, m_p99 = single_server_ramp_stats(lib_mono, dt, n_mc_check, n_steps, rng)
    c_mean, c_max, c_p99 = single_server_ramp_stats(lib_chunk, dt, n_mc_check, n_steps, rng)
    print(f"  mono:      mean={m_mean:.1f} max={m_max:.1f} p99={m_p99:.1f} W/s "
          f"(real measured: mean=46.9 max=2146 p99=856.6)")
    print(f"  chunk=512: mean={c_mean:.1f} max={c_max:.1f} p99={c_p99:.1f} W/s "
          f"(real measured: mean=27.6 max=3763 p99=737.1)")

    print()
    print("=== Step 3: fleet-scale ramp-CF (s=0, fully independent) N-sweep ===")
    n_mc_cf = 30
    for N in [1000, 5000, 10000, 20000]:
        cf_mono = ramp_cf_bootstrap(lib_mono, N, 0.0, dt, n_steps, m_mean, n_mc_cf, rng)
        cf_chunk = ramp_cf_bootstrap(lib_chunk, N, 0.0, dt, n_steps, c_mean, n_mc_cf, rng)
        print(f"  N={N:<6} CF_ramp mono={cf_mono:.4f}  chunk=512={cf_chunk:.4f}  "
              f"reduction={100*(1-cf_chunk/cf_mono):.1f}%")

    print()
    print("=== Step 4: correlation sensitivity at N=10000 (s-sweep, mirrors the level-CF s-sweep) ===")
    N = 10000
    for s in [0.0, 0.01, 0.02, 0.05, 0.1]:
        cf_mono = ramp_cf_bootstrap(lib_mono, N, s, dt, n_steps, m_mean, n_mc_cf, rng)
        cf_chunk = ramp_cf_bootstrap(lib_chunk, N, s, dt, n_steps, c_mean, n_mc_cf, rng)
        print(f"  s={s:<5} CF_ramp mono={cf_mono:.4f}  chunk=512={cf_chunk:.4f}  "
              f"reduction={100*(1-cf_chunk/cf_mono):.1f}%")


if __name__ == "__main__":
    main()
