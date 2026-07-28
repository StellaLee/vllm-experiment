#!/usr/bin/env python3
"""Data-center-scale aggregate power profile: a single, real realization of the validated
two-state Monte Carlo model (coincidence_factor_model.simulate_states + smoothed_aggregate) at
N=10,000 -- the same fleet size used for the reserve-procurement calculation (Table I) -- so
readers can see what the aggregate power trace the rest of Section V/VI is built from actually
looks like, not just its summary statistics (CF vs N, CF vs s).

Calibration matches the rest of the main body exactly: trial-1 wide/closed-loop trace for
P_b/P_max/p_duty/lambda, R=45.7 W/s (mono) / 30.8 W/s (chunk=512) for tau (the 3-trial-averaged
whole-trace ramp rates already cited in Section V's text).

Two panels, sharing the same time axis:
  (a) Level -- the y-axis is necessarily zoomed: at N=10,000 the profile sits within a fraction
      of a percent of its mean, a vanishing sliver of the full N*P_b-to-N*P_max range. Mono and
      chunk=512 track each other almost exactly here, because occupancy (what sets the level)
      is matched by calibration -- this panel alone does NOT show the ramp-rate benefit.
  (b) Ramp rate dP/dt -- the actual differentiated signal the paper's ~29% claim is about.
      Chunk=512's oscillation is visibly smaller in amplitude throughout, even though its level
      in (a) is nearly indistinguishable from mono's.

Usage:
    python scripts/plot_datacenter_power_profile.py --out paper-pes-im/figs/fig_datacenter_power_profile.png
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate, simulate_states, smoothed_aggregate

MONO = "#4c72b0"
CHUNK = "#dd8452"
INK = "#333333"

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--out", default="/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_datacenter_power_profile.png")
    args = ap.parse_args()

    cal = calibrate(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                     "trial 1 (widened, closed-loop)", whale_thresh=15000)
    tau_mono = (cal["P_max"] - cal["P_b"]) / 45.7
    tau_chunk = (cal["P_max"] - cal["P_b"]) / 30.8
    T_window, dt, warmup = 100.0, 0.1, 50.0
    N = args.N

    rng = np.random.default_rng(20260901)
    state = simulate_states(N, 0.0, cal, T_window + warmup, dt, rng)
    D_mono_w = smoothed_aggregate(state, cal["P_b"], cal["P_max"], dt, tau_mono)
    D_chunk_w = smoothed_aggregate(state, cal["P_b"], cal["P_max"], dt, tau_chunk)
    i0 = int(warmup / dt)
    D_mono_w = D_mono_w[i0:]      # W, for ramp calc
    D_chunk_w = D_chunk_w[i0:]

    t = np.arange(D_mono_w.shape[0]) * dt
    D_mono = D_mono_w / 1e6  # W -> MW
    D_chunk = D_chunk_w / 1e6

    ramp_mono = np.diff(D_mono_w) / dt / 1000.0    # W/s -> kW/s
    ramp_chunk = np.diff(D_chunk_w) / dt / 1000.0
    t_ramp = t[1:]

    N_Pb = N * cal["P_b"] / 1e6
    N_Pmax = N * cal["P_max"] / 1e6
    full_range = N_Pmax - N_Pb
    shown_lo = min(D_mono.min(), D_chunk.min())
    shown_hi = max(D_mono.max(), D_chunk.max())
    pad = 0.15 * (shown_hi - shown_lo)
    shown_frac = (shown_hi - shown_lo) / full_range * 100

    max_ramp_mono = np.abs(ramp_mono).max() * 1000  # back to W/s for stats
    max_ramp_chunk = np.abs(ramp_chunk).max() * 1000
    mean_ramp_mono = np.abs(ramp_mono).mean() * 1000
    mean_ramp_chunk = np.abs(ramp_chunk).mean() * 1000
    max_reduction = (1 - max_ramp_chunk / max_ramp_mono) * 100
    mean_reduction = (1 - mean_ramp_chunk / mean_ramp_mono) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.2, 4.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 0.85], "hspace": 0.45})

    ax1.plot(t, D_mono, color=MONO, linewidth=1.2)
    ax1.plot(t, D_chunk, color=CHUNK, linewidth=1.2)
    ax1.text(0.03, 0.90, "mono", color=MONO, fontsize=9.5, weight="bold", transform=ax1.transAxes)
    ax1.text(0.03, 0.06, "chunk=512", color=CHUNK, fontsize=9.5, weight="bold",
             transform=ax1.transAxes)
    ax1.set_ylim(shown_lo - pad, shown_hi + pad)
    ax1.set_ylabel("aggregate power\n(MW)", fontsize=9.5)
    ax1.set_title(f"(a) Level: zoomed to $\\approx${shown_frac:.2f}% of the full range -- "
                  f"nearly identical", fontsize=9.3)
    ax1.tick_params(labelsize=8.5)

    ax2.plot(t_ramp, ramp_mono, color=MONO, linewidth=0.7, alpha=0.9)
    ax2.plot(t_ramp, ramp_chunk, color=CHUNK, linewidth=0.7, alpha=0.9)
    ax2.axhline(0, color=INK, linewidth=0.5, alpha=0.4)
    ax2.set_ylabel("ramp rate $dP/dt$\n(kW/s)", fontsize=9.5)
    ax2.set_xlabel("time (s)", fontsize=9.5)
    ax2.set_title(f"(b) Ramp rate: chunk=512 visibly damped "
                  f"(max {max_reduction:.0f}% lower, mean {mean_reduction:.0f}% lower)",
                  fontsize=9.3)
    ax2.tick_params(labelsize=8.5)
    ax2.set_xlim(0, t[-1])

    fig.suptitle(f"Simulated data-center-scale aggregate power profile ($N{{=}}{N:,}$)",
                 fontsize=10.5, y=0.99)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(f"N={N}: mono max|ramp|={max_ramp_mono:.1f} W/s mean={mean_ramp_mono:.1f} W/s")
    print(f"       chunk=512 max|ramp|={max_ramp_chunk:.1f} W/s mean={mean_ramp_chunk:.1f} W/s")
    print(f"       max reduction={max_reduction:.1f}%  mean reduction={mean_reduction:.1f}%")
    print(f"shown y-range (level) is {shown_frac:.3f}% of full N*(P_max-P_b)={full_range:.4f} MW")


if __name__ == "__main__":
    main()
