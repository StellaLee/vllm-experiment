#!/usr/bin/env python3
"""Data-center-scale aggregate power profile: a single, real realization of the CORRECTED
Monte Carlo model (per-policy real timing, from cf_continuous_regime_model.calibrate_continuous
+ ou_continuous_aggregate) at N=10,000 -- the same fleet size used for the reserve-procurement
calculation (Table I) -- so readers can see what the aggregate power trace the rest of
Section V/VI is built from actually looks like, not just its summary statistics (CF vs N, CF vs s).

SUPERSEDES an earlier version of this figure that shared ONE calibration (P_b, P_max, duty
cycle, mean burst duration -- all from mono's trace only) between mono and chunk, varying only
the smoothing time constant tau, and used a flat (noise-free) power level within each regime.
Both assumptions were checked against measurement and found wrong: chunk's own trace gives a
meaningfully different duty cycle (0.696 vs mono's 0.561) and burst duration (3.68s vs 1.63s),
and real "baseline" power is not flat -- it fluctuates continuously (std ~37-40W) even outside
whale windows. This version calibrates mono and chunk SEPARATELY from their own traces, and
models within-regime power as an Ornstein-Uhlenbeck process (mean-reverting, real calibrated
variance) rather than a constant. The corrected model's reserve-procurement numbers land within
a point of the original (Table~\\ref{tab:reserve}), so the headline is not materially changed --
but the methodology is now free of two identified, falsifiable-and-falsified assumptions.

Two panels, sharing the same time axis:
  (a) Level -- the y-axis is necessarily zoomed. Because occupancy is NOT matched between
      policies in this corrected model (chunk's duty cycle is genuinely higher), chunk's level
      sits measurably above mono's here -- unlike the earlier (incorrect) figure, which showed
      them as indistinguishable by construction.
  (b) Ramp rate dP/dt -- the differentiated signal the paper's reserve-procurement claim is
      about. Chunk=512's oscillation is visibly smaller in amplitude throughout.

Usage:
    python scripts/plot_datacenter_power_profile.py --out paper-pes-im/figs/fig_datacenter_power_profile.png
"""
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous, ou_continuous_aggregate
from coincidence_factor_model import simulate_states

MONO = "#4c72b0"
CHUNK = "#dd8452"
INK = "#333333"

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10000)
    ap.add_argument("--out", default="/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_datacenter_power_profile.png")
    args = ap.parse_args()

    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv",
                                     "mono/16384 (own trace)")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv",
                                      "chunk=512 (own trace)")
    kappa_mono, kappa_chunk = 0.1320, 0.0710  # fit so simulated mean|dP/dt| matches measured 45.7/30.8 W/s
    T_window, dt, warmup = 100.0, 0.1, 50.0
    N = args.N

    rng = np.random.default_rng(20260901)
    state_mono = simulate_states(N, 0.0, cal_mono, T_window + warmup, dt, rng)
    state_chunk = simulate_states(N, 0.0, cal_chunk, T_window + warmup, dt, rng)
    P_mono = ou_continuous_aggregate(state_mono, cal_mono["mu_b"], cal_mono["mu_max"],
                                      cal_mono["sigma_b"], cal_mono["sigma_max"], dt, kappa_mono, rng)
    P_chunk = ou_continuous_aggregate(state_chunk, cal_chunk["mu_b"], cal_chunk["mu_max"],
                                       cal_chunk["sigma_b"], cal_chunk["sigma_max"], dt, kappa_chunk, rng)
    D_mono_w = P_mono.sum(axis=0)
    D_chunk_w = P_chunk.sum(axis=0)
    i0 = int(warmup / dt)
    D_mono_w = D_mono_w[i0:]      # W, for ramp calc
    D_chunk_w = D_chunk_w[i0:]

    t = np.arange(D_mono_w.shape[0]) * dt
    D_mono = D_mono_w / 1e6  # W -> MW
    D_chunk = D_chunk_w / 1e6

    ramp_mono = np.diff(D_mono_w) / dt / 1000.0    # W/s -> kW/s
    ramp_chunk = np.diff(D_chunk_w) / dt / 1000.0
    t_ramp = t[1:]

    N_Pmax_mono = N * cal_mono["mu_max"] / 1e6
    N_Pb_mono = N * cal_mono["mu_b"] / 1e6
    full_range = N_Pmax_mono - N_Pb_mono
    shown_lo = min(D_mono.min(), D_chunk.min())
    shown_hi = max(D_mono.max(), D_chunk.max())
    pad = 0.15 * (shown_hi - shown_lo)
    shown_frac = (shown_hi - shown_lo) / full_range * 100
    level_diff_pct = (D_chunk.mean() / D_mono.mean() - 1) * 100

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
    ax1.text(0.03, 0.06, "mono", color=MONO, fontsize=9.5, weight="bold", transform=ax1.transAxes)
    ax1.text(0.03, 0.90, "chunk=512", color=CHUNK, fontsize=9.5, weight="bold",
             transform=ax1.transAxes)
    ax1.set_ylim(shown_lo - pad, shown_hi + pad)
    ax1.set_ylabel("aggregate power\n(MW)", fontsize=9.5)
    ax1.set_title(f"(a) Level: zoomed to $\\approx${shown_frac:.2f}% of the full range -- "
                  f"chunk $\\approx${level_diff_pct:.0f}% higher (unequal duty cycle)", fontsize=9.3)
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
    print(f"       level: chunk {level_diff_pct:+.1f}% vs mono (duty cycle {cal_chunk['p_duty']:.3f} vs {cal_mono['p_duty']:.3f})")
    print(f"shown y-range (level) is {shown_frac:.3f}% of full N*(P_max-P_b)={full_range:.4f} MW (mono ref)")


if __name__ == "__main__":
    main()
