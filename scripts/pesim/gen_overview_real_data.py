#!/usr/bin/env python3
"""Generates the REAL (measured + simulated) data behind the paper's overview schematic
(fig_overview_schematic.png / plot_paper_overview_schematic.py). Nothing here is fabricated:

(a) Single-GPU panel: an actual excerpt of the measured power trace (trial 1, wide/closed-loop
    condition), around the SAME replayed whale request (idx=2 in the request log, pad_chars
    34834, identical across arms because all arms replay the same seeded request sequence) for
    mono/16384 and chunk=512 -- directly showing the same underlying prompt processed two ways.

(b) Fleet-aggregate panel: a real (small-N, illustrative) run of the validated two-state
    Monte Carlo model in coincidence_factor_model.py (simulate_states + a per-server variant of
    smoothed_aggregate that keeps the per-server traces instead of only their sum), calibrated
    from the same trial-1 trace, at s=0. N is deliberately small (25) so individual traces are
    visually distinguishable -- the paper's actual reserve numbers (panel c) use N=10,000.

(c) Reserve-procurement panel: the exact Table-I/Table-II methodology
    (cf_main_wide_wholetrace.py), i.e. collect_peak_ramps at N=10,000 pooled across all 3 real
    trials, giving the same 29-31% headline reduction reported in the paper.

Usage:
    python scripts/gen_overview_real_data.py --out scripts/overview_real_data.npz
"""
import argparse
import sys

import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate, load_power, load_records, simulate_states
from ramp_reserve_procurement import collect_peak_ramps

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"

TRIALS = [
    dict(t=1, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
         mono_ramp=46.0, chunk_ramp=27.8),
    dict(t=2, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t2-power.csv",
         mono_ramp=46.5, chunk_ramp=34.3),
    dict(t=3, rec=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3.jsonl",
         pw=f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t3-power.csv",
         mono_ramp=44.6, chunk_ramp=30.2),
]

WHALE_IDX = 2  # pad_chars=34834, identical across arms (same replayed request sequence)


def rolling_mean(x, win):
    """Light smoothing of raw ~50ms NVML samples for display -- removes meter/sampling jitter,
    does not alter the underlying whale-driven envelope (win=5 samples =~ 0.25s at 50ms)."""
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    pad = win // 2
    xp = np.pad(x, (pad, win - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def panel_a_data():
    """Real measured power around the same replayed whale request, mono vs chunk=512."""
    out = {}
    for arm, tag, post_pad in [("b16384", "mono", 3.0), ("b512", "chunk", 3.0)]:
        rec_path = f"{RAW}/2026-07-26-pesim_gate_wide-{arm}-t1.jsonl"
        pw_path = f"{RAW}/2026-07-26-pesim_gate_wide-{arm}-t1-power.csv"
        recs = load_records(rec_path)
        power = load_power(pw_path)
        r = recs[WHALE_IDX]
        assert r["pad_chars"] >= 15000, r["pad_chars"]
        whale_start = r["ts"] - r["latency"]
        whale_end = r["ts"]
        t = np.array([p[0] for p in power])
        pw = np.array([p[1] for p in power])
        mask = (t >= whale_start - 1.5) & (t <= whale_end + post_pad)
        out[f"{tag}_t"] = t[mask] - whale_start
        out[f"{tag}_p"] = rolling_mean(pw[mask], 5)
        out[f"{tag}_whale_dur"] = whale_end - whale_start
        print(f"panel (a) {tag}: whale duration {whale_end - whale_start:.2f}s, "
              f"{mask.sum()} power samples in display window")
    return out


def smoothed_matrix(state, P_b, P_max, dt, tau):
    """Same recursion as coincidence_factor_model.smoothed_aggregate, but returns the
    per-server (N, n_steps) trace instead of collapsing it with .sum(axis=0) -- needed so the
    illustration panel can show individual faint server traces alongside the bold aggregate."""
    N, n_steps = state.shape
    target = np.where(state, P_max, P_b).astype(float)
    P = np.empty_like(target)
    P[:, 0] = P_b
    alpha = dt / tau
    for i in range(1, n_steps):
        P[:, i] = P[:, i - 1] + alpha * (target[:, i - 1] - P[:, i - 1])
    return P


def panel_b_data():
    """Real (small-N, illustrative) run of the validated Monte Carlo model."""
    tr = TRIALS[0]
    cal = calibrate(tr["rec"], tr["pw"], "trial 1 (panel b calibration)", whale_thresh=15000)
    tau_mono = (cal["P_max"] - cal["P_b"]) / tr["mono_ramp"]
    tau_chunk = (cal["P_max"] - cal["P_b"]) / tr["chunk_ramp"]

    N, dt, warmup, T_display = 6, 0.1, 30.0, 25.0
    rng = np.random.default_rng(7)
    state = simulate_states(N, 0.0, cal, T_display + warmup, dt, rng)
    i0 = int(warmup / dt)

    P_mono = smoothed_matrix(state, cal["P_b"], cal["P_max"], dt, tau_mono)[:, i0:]
    P_chunk = smoothed_matrix(state, cal["P_b"], cal["P_max"], dt, tau_chunk)[:, i0:]
    t = np.arange(P_mono.shape[1]) * dt

    print(f"panel (b): N={N} servers, {P_mono.shape[1]} steps post-warmup, "
          f"mono mean={P_mono.mean(axis=0).mean():.1f}W chunk mean={P_chunk.mean(axis=0).mean():.1f}W")
    return dict(b_t=t, b_mono=P_mono, b_chunk=P_chunk, b_N=N)


def panel_c_data():
    """Exact Table-I methodology, pooled across all 3 real trials -- reproduces the paper's
    29-31% headline reserve reduction."""
    N, n_mc = 10000, 300
    T_window, dt, warmup = 100.0, 0.1, 50.0
    mono_pool, chunk_pool = [], []
    r99_mono_trials, r99_chunk_trials = [], []

    for tr in TRIALS:
        label = f"trial {tr['t']}"
        cal = calibrate(tr["rec"], tr["pw"], label, whale_thresh=15000)
        tau_mono = (cal["P_max"] - cal["P_b"]) / tr["mono_ramp"]
        tau_chunk = (cal["P_max"] - cal["P_b"]) / tr["chunk_ramp"]
        rng = np.random.default_rng(20260807 + tr["t"])

        mono_peaks = collect_peak_ramps(N, 0.0, cal, tau_mono, T_window, dt, n_mc, rng, warmup)
        chunk_peaks = collect_peak_ramps(N, 0.0, cal, tau_chunk, T_window, dt, n_mc, rng, warmup)
        mono_mw = mono_peaks * 60.0 / 1e6
        chunk_mw = chunk_peaks * 60.0 / 1e6
        mono_pool.append(mono_mw)
        chunk_pool.append(chunk_mw)
        r99_mono_trials.append(np.percentile(mono_mw, 99))
        r99_chunk_trials.append(np.percentile(chunk_mw, 99))
        print(f"panel (c) {label}: mono R99={r99_mono_trials[-1]:.4f} "
              f"chunk=512 R99={r99_chunk_trials[-1]:.4f} MW/min")

    mono_pool = np.concatenate(mono_pool)
    chunk_pool = np.concatenate(chunk_pool)
    r99_mono = float(np.mean(r99_mono_trials))
    r99_chunk = float(np.mean(r99_chunk_trials))
    print(f"panel (c) SUMMARY (mean of 3 per-trial R99, matches paper Table I): "
          f"mono={r99_mono:.4f} chunk=512={r99_chunk:.4f} MW/min "
          f"reduction={(1 - r99_chunk / r99_mono) * 100:.1f}%")
    return dict(c_mono_pool=mono_pool, c_chunk_pool=chunk_pool,
                c_r99_mono=r99_mono, c_r99_chunk=r99_chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/Users/li/Documents/vllm-experiment/scripts/overview_real_data.npz")
    ap.add_argument("--skip-c", action="store_true",
                    help="reuse panel (c)'s cached arrays from --out (fast iteration on a/b)")
    args = ap.parse_args()

    data = {}
    if args.skip_c:
        old = np.load(args.out)
        data.update({k: old[k] for k in old.files if k.startswith("c_")})
    data.update(panel_a_data())
    data.update(panel_b_data())
    if not args.skip_c:
        data.update(panel_c_data())
    np.savez(args.out, **data)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
