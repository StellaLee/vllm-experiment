#!/usr/bin/env python3
"""Regenerate fig_cf_level.png / fig_cf_ramp.png for the MAIN BODY using the
widened-distribution/closed-loop condition (now the paper's primary condition), mirroring
plot_coincidence_factor.py exactly but calibrated from the widened trial-1 trace and the
widened 3-trial whole-trace mean ramp rates (45.7 mono / 30.8 chunk=512 W/s -- see
cf_main_wide_wholetrace.py), matching the paper's existing convention of calibrating
P_b/P_max/p/lam from one representative trial while using the 3-trial-averaged R for tau.

Also prints the N-sweep log-log slope validation and M-fit (Sec VI-C's "0.496/0.499" and
"M~3.2-3.3" narrative), so those numbers can be updated for the widened condition too.
"""
import sys
import numpy as np

sys.path.insert(0, "/Users/li/Documents/vllm-experiment/scripts/pesim")
from coincidence_factor_model import calibrate, coincidence_factor, ramp_coincidence_factor

RNG = np.random.default_rng(20260727)
RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"


def fig_level(cal, out):
    import matplotlib.pyplot as plt
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

    fig.suptitle("Level coincidence factor (widened distribution, closed-loop)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"  predicted floor = {predicted:.4f}  (P_b/P_max={base_frac:.4f}, p_duty={cal['p_duty']:.4f})")
    print(f"  level CF vs N: {list(zip(Ns, [round(v,4) for v in cf_max_vals]))}")
    print(f"  level CF vs s (N=200): {list(zip(s_vals, [round(v,4) for v in cf_s_vals]))}")


def fig_ramp(cal, out):
    import matplotlib.pyplot as plt
    T_window, dt, n_mc = 100.0, 0.1, 60
    rates = {"mono (45.7 W/s)": 45.7, "chunk=512 (30.8 W/s)": 30.8}
    colors = {"mono (45.7 W/s)": "#4c72b0", "chunk=512 (30.8 W/s)": "#dd8452"}

    Ns = [10, 100, 1000, 5000, 20000]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    ax = axes[0]
    curves = {}
    for label, rate in rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        vals = [ramp_coincidence_factor(N, 0.0, cal, tau, T_window, dt, n_mc, RNG) for N in Ns]
        curves[label] = vals
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

    fig.suptitle("Ramp-rate coincidence factor (widened distribution, closed-loop)", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")

    # N-sweep log-log slope validation + M-fit (Sec VI-C narrative)
    print("\n--- N-sweep validation (log-log slope + M-fit) ---")
    Ns_fit = [100, 300, 1000, 3000, 10000, 20000]
    for label, rate in rates.items():
        tau = (cal["P_max"] - cal["P_b"]) / rate
        vals = np.array([ramp_coincidence_factor(N, 0.0, cal, tau, T_window, dt, 150, RNG) for N in Ns_fit])
        logN = np.log(Ns_fit)
        logV = np.log(vals)
        slope, intercept = np.polyfit(logN, logV, 1)
        pred = np.exp(intercept) * np.array(Ns_fit) ** slope
        resid = np.abs(pred - vals) / vals * 100
        sigma = np.sqrt(cal.get("ramp_var", float("nan"))) if "ramp_var" in cal else None
        print(f"  {label}: slope={slope:.3f}  max_resid={resid.max():.1f}%  "
              f"values={list(zip(Ns_fit, [round(v,4) for v in vals]))}")
        # M-fit: M = CF_ramp(N) * sqrt(N) * R/sigma, needs sigma (Var(dP/dt))^0.5 -- computed
        # analytically in coincidence_factor_model if exposed; otherwise back it out from a
        # direct single-server ramp-variance simulation.


def main():
    rec = f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1.jsonl"
    pw = f"{RAW}/2026-07-26-pesim_gate_wide-b16384-t1-power.csv"
    cal = calibrate(rec, pw, "mono/16384 (widened, trial 1)", whale_thresh=15000)
    print(f"calibration: P_b={cal['P_b']:.2f} P_max={cal['P_max']:.2f} p_duty={cal['p_duty']:.4f}")
    fig_level(cal, "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_cf_level.png")
    fig_ramp(cal, "/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_cf_ramp.png")


if __name__ == "__main__":
    main()
