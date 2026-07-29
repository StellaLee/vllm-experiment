#!/usr/bin/env python3
"""Validation figure for the per-policy-timing + continuous-OU model: (a) real vs. one
simulated realization's power trace snippet, for mono and chunk=512; (b) real vs. simulated
single-server ramp-rate CCDF (log-y), showing where the fit is good (bulk/mean, what was
calibrated) and where it is not (tail -- disclosed honestly, not hidden)."""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from cf_continuous_regime_model import calibrate_continuous, ou_continuous_aggregate
from coincidence_factor_model import simulate_states, load_power

RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
MONO = "#4c72b0"
CHUNK = "#dd8452"


def real_ramp(power_csv, trim=5.0):
    power = load_power(power_csv)
    t0, t1 = power[0][0], power[-1][0]
    arr = np.array(power)
    m = (arr[:, 0] >= t0 + trim) & (arr[:, 0] <= t1 - trim)
    arr = arr[m]
    return arr[:, 0] - arr[0, 0], arr[:, 1], np.abs(np.diff(arr[:, 1])) / np.diff(arr[:, 0])


def sim_trace_and_ramp(cal, kappa, dt, T, rng):
    state = simulate_states(1, 0.0, cal, T, dt, rng)
    P = ou_continuous_aggregate(state, cal["mu_b"], cal["mu_max"], cal["sigma_b"],
                                 cal["sigma_max"], dt, kappa, rng)[0]
    t = np.arange(len(P)) * dt
    ramp = np.abs(np.diff(P)) / dt
    return t, P, ramp


def ccdf(x):
    xs = np.sort(x)
    return xs, 1.0 - np.arange(1, len(xs) + 1) / len(xs)


def main():
    dt = 0.1
    rng = np.random.default_rng(42)
    cal_mono = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl",
                                     f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv", "mono")
    cal_chunk = calibrate_continuous(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1.jsonl",
                                      f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv", "chunk=512")
    kappa_mono, kappa_chunk = 0.1320, 0.0710

    t_real_m, p_real_m, ramp_real_m = real_ramp(f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv")
    t_real_c, p_real_c, ramp_real_c = real_ramp(f"{RAW}/2026-07-26-pesim_gate_wide-b512-t1-power.csv")
    T_sim = min(t_real_m[-1], t_real_c[-1])

    t_sim_m, p_sim_m, ramp_sim_m = sim_trace_and_ramp(cal_mono, kappa_mono, dt, T_sim, rng)
    t_sim_c, p_sim_c, ramp_sim_c = sim_trace_and_ramp(cal_chunk, kappa_chunk, dt, T_sim, rng)

    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.7))

    for ax, ramp_real, ramp_sim, color, label in [
        (axes[0], ramp_real_m, ramp_sim_m, MONO, "mono"),
        (axes[1], ramp_real_c, ramp_sim_c, CHUNK, "chunk=512"),
    ]:
        xr, yr = ccdf(ramp_real)
        xs, ys = ccdf(ramp_sim)
        ax.loglog(xr, yr, color=color, lw=1.5, label="real")
        ax.loglog(xs, ys, color="gray", lw=1.5, alpha=0.8, label="simulated (OU)")
        ax.set_title(f"{label}", fontsize=9.5)
        ax.set_xlabel("$|dP/dt|$ (W/s)", fontsize=9); ax.set_ylabel("$P(\\mathrm{ramp}>x)$", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7.5, loc="lower left")
        ax.annotate("mean\nmatches", xy=(ramp_real.mean(), 0.55), fontsize=6.5,
                    color="green", ha="center")
        ax.annotate("tail\ndiverges", xy=(np.percentile(ramp_real, 99.7), 0.008),
                    fontsize=6.5, color="red", ha="center")

    fig.suptitle("Single-server ramp-rate: real vs. simulated (OU model)", fontsize=10)
    plt.tight_layout()
    out = "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_ou_validation.png"
    plt.savefig(out, dpi=200)
    print(f"wrote {out}")
    print(f"\nmono:   real mean_ramp={ramp_real_m.mean():.1f}  sim mean_ramp={ramp_sim_m.mean():.1f}"
          f"  | real p99={np.percentile(ramp_real_m,99):.1f}  sim p99={np.percentile(ramp_sim_m,99):.1f}")
    print(f"chunk:  real mean_ramp={ramp_real_c.mean():.1f}  sim mean_ramp={ramp_sim_c.mean():.1f}"
          f"  | real p99={np.percentile(ramp_real_c,99):.1f}  sim p99={np.percentile(ramp_sim_c,99):.1f}")


if __name__ == "__main__":
    main()
