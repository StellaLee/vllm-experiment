#!/usr/bin/env python3
"""Generate the two coincidence-factor figures for paper.md/main.tex Sec 6:
  1. fig_cf_level.png: (a) level CF vs N at s=0, converging to the baseline-corrected floor;
     (b) level CF vs s at fixed N=200, showing the threshold jump.
  2. fig_cf_ramp.png: (a) ramp-rate CF vs N at s=0 for mono/chunk=512 tau, showing the
     fleet-scale ramp-coincidence risk shrinking as ~1/sqrt(N) for both policies (it
     averages out, unlike the level metric's floor); (b) ramp-rate CF vs s at N=200,
     contrasted with the level metric's threshold to show it rises smoothly instead.

Run from the directory containing mono_t1_records.jsonl / mono_t1_power.csv (see
coincidence_factor_model.py's own docstring for how to obtain them).
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts")
from coincidence_factor_model import calibrate, coincidence_factor, ramp_coincidence_factor

RNG = np.random.default_rng(20260726)


def fig_level(cal, out):
    T_window, dt, n_mc = 100.0, 0.1, 200
    base_frac = cal["P_b"] / cal["P_max"]
    predicted = base_frac + (1 - base_frac) * cal["p_duty"]

    Ns = [10, 30, 100, 300, 1000]
    cf_max_vals, cf_p99_vals = [], []
    for N in Ns:
        cf_max, cf_p99 = coincidence_factor(N, 0.0, cal, T_window, dt, n_mc, RNG)
        cf_max_vals.append(cf_max)
        cf_p99_vals.append(cf_p99)

    s_vals = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.25, 0.5]
    cf_s_vals = [coincidence_factor(200, s, cal, T_window, dt, n_mc, RNG)[0] for s in s_vals]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ax.semilogx(Ns, cf_max_vals, "o-", color="#4c72b0", label=r"$CF_{max}$")
    ax.semilogx(Ns, cf_p99_vals, "s--", color="#55a868", label=r"$CF_{p99}$")
    ax.axhline(predicted, color="#c44e52", linestyle=":", linewidth=1.2,
               label=f"predicted floor ({predicted:.3f})")
    ax.set_xlabel("N (fleet size)")
    ax.set_ylabel("coincidence factor")
    ax.set_title("(a) vs. N, s=0", fontsize=10)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0.9, 1.005)

    ax = axes[1]
    naive = [cf_s_vals[0] + s * (1.0 - cf_s_vals[0]) for s in s_vals]
    ax.plot(s_vals, naive, "--", color="#999999", linewidth=1.2, label="naive linear interp.")
    ax.plot(s_vals, cf_s_vals, "o-", color="#4c72b0", label="simulated")
    ax.axvspan(0.05, 0.10, color="#c44e52", alpha=0.15)
    ax.set_xlabel("s (correlated-arrival probability)")
    ax.set_ylabel(r"$CF_{max}$ (N=200)")
    ax.set_title("(b) vs. s, N=200", fontsize=10)
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Level coincidence factor", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  level CF vs N: {list(zip(Ns, [round(v,4) for v in cf_max_vals]))}")
    print(f"  level CF vs s (N=200): {list(zip(s_vals, [round(v,4) for v in cf_s_vals]))}")


def fig_ramp(cal, out):
    T_window, dt, n_mc = 100.0, 0.1, 60
    rates = {"mono (41.3 W/s)": 41.3, "chunk=512 (27.1 W/s)": 27.1}
    colors = {"mono (41.3 W/s)": "#4c72b0", "chunk=512 (27.1 W/s)": "#dd8452"}

    Ns = [10, 100, 1000, 5000, 20000]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    for label, rate in rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        vals = [ramp_coincidence_factor(N, 0.0, cal, tau, T_window, dt, n_mc, RNG) for N in Ns]
        ax.semilogx(Ns, vals, "o-", color=colors[label], label=label)
    ax.set_xlabel("N (fleet size)")
    ax.set_ylabel(r"$CF_{ramp}$ (s=0)")
    ax.set_title(r"(a) shrinks $\sim 1/\sqrt{N}$, doesn't saturate", fontsize=10)
    ax.legend(fontsize=7)

    s_vals = [0.0, 0.02, 0.05, 0.1, 0.15, 0.25, 0.5]
    ax = axes[1]
    for label, rate in rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        vals = [ramp_coincidence_factor(200, s, cal, tau, T_window, dt, n_mc, RNG) for s in s_vals]
        ax.plot(s_vals, vals, "o-", color=colors[label], label=label)
    ax.set_xlabel("s (correlated-arrival probability)")
    ax.set_ylabel(r"$CF_{ramp}$ (N=200)")
    ax.set_title("(b) rises smoothly, no threshold", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("Ramp-rate coincidence factor", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


def main():
    cal = calibrate("mono_t1_records.jsonl", "mono_t1_power.csv", "mono/16384")
    fig_level(cal, "fig_cf_level.png")
    fig_ramp(cal, "fig_cf_ramp.png")


if __name__ == "__main__":
    main()
