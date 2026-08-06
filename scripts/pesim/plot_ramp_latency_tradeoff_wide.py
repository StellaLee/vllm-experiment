#!/usr/bin/env python3
"""fig_ramp_latency_tradeoff.png regenerated for the widened-distribution/closed-loop
condition (now the paper's main-body condition) -- same structure as
plot_ramp_latency_tradeoff.py, applied to data/pesim_gate_raw/2026-07-26-pesim_gate_wide-*
with whale_thresh=15000 (the widened whale floor, vs. the narrow trace's 40000) and the
widened whole-trace ramp means (45.7/42.8/30.8 W/s, from analyze_power_shape.py).
"""
import json
import statistics
import matplotlib.pyplot as plt

WHALE_THRESH = 15000
RAW = "/Users/li/Documents/vllm-experiment/data/pesim_gate_raw"
BUDGETS = ["b16384", "b2048", "b512"]
LABELS = ["mono\n(16384)", "chunk\n(2048)", "chunk\n(512)"]
RAMP_W_S = {"b16384": 45.7, "b2048": 42.8, "b512": 30.8}  # per-trial means, widened Table II


def load(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def pctl(vals, p):
    s = sorted(vals)
    k = int(round(p / 100 * (len(s) - 1)))
    return s[k]


def main():
    whale_ttft_mean, whale_ttft_p99 = [], []
    short_tbt_p99 = []
    for b in BUDGETS:
        wt, st = [], []
        for tr in [1, 2, 3]:
            recs = load(f"{RAW}/2026-07-26-pesim_gate_wide-{b}-t{tr}.jsonl")
            for r in recs:
                pc = r.get("pad_chars", 0) or 0
                if pc >= WHALE_THRESH:
                    if r.get("ttft") is not None:
                        wt.append(r["ttft"] * 1000)
                else:
                    st.extend(r.get("tbt_ms") or [])
        whale_ttft_mean.append(statistics.mean(wt))
        whale_ttft_p99.append(pctl(wt, 99))
        short_tbt_p99.append(pctl(st, 99))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)

    ax = axes[0]
    ramp_vals = [RAMP_W_S[b] for b in BUDGETS]
    l1, = ax.plot(LABELS, ramp_vals, "o-", color="#4c72b0", label="ramp rate (W/s)")
    ax.set_ylabel("mean ramp rate (W/s)", color="#4c72b0", fontsize=8.5)
    ax.tick_params(axis="y", labelcolor="#4c72b0", labelsize=8)
    ax.tick_params(axis="x", labelsize=8)
    ax2 = ax.twinx()
    l2, = ax2.plot(LABELS, short_tbt_p99, "s--", color="#55a868", label="short-req TBT p99 (ms)")
    ax2.set_ylabel("short-request TBT p99 (ms)", color="#55a868", fontsize=8.5)
    ax2.tick_params(axis="y", labelcolor="#55a868", labelsize=8)
    ax.set_title("(a) Power smoothing tracks short-req\ntail-latency protection", fontsize=10)
    ax.legend(handles=[l1, l2], fontsize=7, loc="center left")

    ax = axes[1]
    ax.plot(LABELS, whale_ttft_mean, "o-", color="#c44e52", label="whale TTFT mean")
    ax.plot(LABELS, whale_ttft_p99, "s--", color="#dd8452", label="whale TTFT p99")
    ax.set_ylabel("whale TTFT (ms)", fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title("(b) ...at a whale-TTFT cost", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("Ramp-rate benefit vs. latency (widened distribution, closed-loop)", fontsize=10.5)
    fig.savefig("/Users/li/Documents/vllm-experiment/paper-pes-im/figs/fig_ramp_latency_tradeoff.png", dpi=200, bbox_inches="tight")
    print("wrote fig_ramp_latency_tradeoff.png")
    print("whale_ttft_mean:", list(zip(BUDGETS, [round(v) for v in whale_ttft_mean])))
    print("whale_ttft_p99:", list(zip(BUDGETS, [round(v) for v in whale_ttft_p99])))
    print("short_tbt_p99:", list(zip(BUDGETS, [round(v, 1) for v in short_tbt_p99])))


if __name__ == "__main__":
    main()
