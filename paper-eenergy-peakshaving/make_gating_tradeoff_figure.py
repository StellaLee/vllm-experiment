"""Generates figs/gating_tradeoff.pdf: a 2x2 grid -- closed-loop (top row) and Poisson
(bottom row) each get their own (mean TTFT, max windowed-average peak power) and (mean TPOT,
max windowed-average peak power) panels, with independent axis scaling per row (Poisson's
TTFT range is 5-10x closed-loop's, so sharing axes squashes the closed-loop trend). One line
per pool (prefill/decode) across the 4 intensities in each row -- see
findings/2026-09-11-eenergy-disagg-peak-shaving-gate.md and
paper-eenergy-peakshaving/paper.md Sec 5.4-5.5 for the underlying numbers.

Source data: printed by scripts/eenergy/analyze_disagg_8gpu_power.py against each condition's
records.jsonl/power_*.csv on the 8x4090 server (183.147.142.123:/root/pli/vllm-experiment/
logs/), embedded below as the final summary statistics. Re-run analyze_disagg_8gpu_power.py
per condition on the remote box and update the dicts below to regenerate with new data.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs", "gating_tradeoff.pdf")

# Closed-loop gating-intensity sweep (concurrency=32, num-convs=600, Heavy/CL-long-shaped
# workload). intensity = cap as a fraction of that pool's own uncapped sustained mean power
# (None = no gate). ttft/tpot are fleet-wide (request-level, not per-pool -- a single
# request's TTFT/TPOT reflects both legs together); power is that pool's own max
# windowed-average (15s). no-gate and 92.5% points are n=3 means (power, ttft); tpot and the
# 80%/65% points remain n=1 (not yet replicated).
CLOSED_LOOP = {
    "prefill": [
        # (intensity_label, mean_ttft_s, mean_tpot_ms, max_windowed_avg_w)
        ("no-gate", 1.770, 18.15, 1121.2),
        ("92.5%", 10.951, 17.57, 919.0),
        ("80%", 13.784, 17.47, 817.7),
        ("65%", 20.913, 17.24, 736.1),
    ],
    "decode": [
        ("no-gate", 1.770, 18.15, 1210.0),
        ("92.5%", 10.951, 17.57, 1187.3),
        ("80%", 13.784, 17.47, 1190.8),
        ("65%", 20.913, 17.24, 1186.8),
    ],
}

# Poisson (open-loop, rate=10.7 conv/s, num-convs=750), same cap values as CLOSED_LOOP's
# 80%/65% rows (same absolute Watts, not re-derived from Poisson's own mean) -- the whole
# point is showing the SAME cap's cost/effect under a different arrival pattern. no-gate and
# 92.5% points are n=3 means; 80%/65% are n=1, freshly measured (2026-09-12): 80% cap ->
# n_failed=84/750 (11.2%); 65% cap -> n_failed=229/750 (30.5%) -- a real, load-bearing
# caveat, not a rounding footnote (see paper Sec 5.5).
POISSON = {
    "prefill": [
        ("no-gate", 14.154, 20.20, 1474.5),
        ("92.5%", 102.856, 17.87, 1115.8),
        ("80%", 98.019, 18.07, 926.8),
        ("65%", 100.395, 17.74, 749.0),
    ],
    "decode": [
        ("no-gate", 14.154, 20.20, 1252.6),
        ("92.5%", 102.856, 17.87, 1192.2),
        ("80%", 98.019, 18.07, 1188.4),
        ("65%", 100.395, 17.74, 1181.6),
    ],
}

POOL_COLOR = {"prefill": "#c0392b", "decode": "#2874a6"}


def _plot_panel(ax, data, x_key_index, x_label, title):
    for pool in ("prefill", "decode"):
        rows = data[pool]
        xs = [row[x_key_index] for row in rows]
        ys = [row[3] for row in rows]
        ax.plot(xs, ys, "o-", color=POOL_COLOR[pool], label=pool, linewidth=2, markersize=6)
        for label, x, y in zip([row[0] for row in rows], xs, ys):
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=6.5,
                        color=POOL_COLOR[pool])
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel("max windowed-avg peak power (W)", fontsize=8.5)
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.2))
    _plot_panel(axes[0, 0], CLOSED_LOOP, 1, "mean TTFT (s)",
                "(a) Closed-loop: TTFT vs. peak power")
    _plot_panel(axes[0, 1], CLOSED_LOOP, 2, "mean TPOT (ms)",
                "(b) Closed-loop: TPOT vs. peak power")
    _plot_panel(axes[1, 0], POISSON, 1, "mean TTFT (s)",
                "(c) Poisson: TTFT vs. peak power")
    _plot_panel(axes[1, 1], POISSON, 2, "mean TPOT (ms)",
                "(d) Poisson: TPOT vs. peak power")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02),
               fontsize=9)
    fig.suptitle("Admission-gate intensity vs. latency vs. peak power, per pool and arrival "
                  "pattern (4:4 P/D-disaggregated, 8 GPUs)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
