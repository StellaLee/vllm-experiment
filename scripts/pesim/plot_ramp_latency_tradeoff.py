#!/usr/bin/env python3
"""Figure: does the ramp-rate/duty-cycle power benefit (Sec V of the paper) come paired with
a latency tradeoff? Uses the same 3-trial gate-experiment dataset as Table II.

Panel (a): ramp rate (from analyze_power_shape.py's per-trial mean, W/s) and short-request
TBT p99 (ms) vs budget, on twin axes -- both fall together as chunking gets more aggressive,
tying the power-shape story to the ALREADY-KNOWN short-request tail-latency protection that
motivates chunked prefill in the first place (Sec II).

Panel (b): whale-request TTFT (mean, p99, ms) vs budget -- the cost side of the same
tradeoff. Non-monotonic: 2048 is actually the best point for TTFT (matches the MLSys-side
longprompt-tbt-win finding that 2048 improves TTFT in every trial while 512's cost is milder
but present), only 512 shows a clear TTFT cost.

Run from the directory containing 2026-07-24-pesimgate-{arm}-t{1,2,3}.jsonl.
"""
import json
import statistics
import matplotlib.pyplot as plt

WHALE_THRESH = 40000
BUDGETS = ["b16384", "b2048", "b512"]
LABELS = ["mono\n(16384)", "chunk\n(2048)", "chunk\n(512)"]
RAMP_W_S = {"b16384": 41.3, "b2048": 39.2, "b512": 27.1}  # per-trial means, Table II


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
            recs = load(f"2026-07-24-pesimgate-{b}-t{tr}.jsonl")
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

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    ax = axes[0]
    ramp_vals = [RAMP_W_S[b] for b in BUDGETS]
    l1, = ax.plot(LABELS, ramp_vals, "o-", color="#4c72b0", label="ramp rate (W/s)")
    ax.set_ylabel("mean ramp rate (W/s)", color="#4c72b0")
    ax.tick_params(axis="y", labelcolor="#4c72b0")
    ax2 = ax.twinx()
    l2, = ax2.plot(LABELS, short_tbt_p99, "s--", color="#55a868", label="short-req TBT p99 (ms)")
    ax2.set_ylabel("short-request TBT p99 (ms)", color="#55a868")
    ax2.tick_params(axis="y", labelcolor="#55a868")
    ax.set_title("(a) Power smoothing tracks short-req\ntail-latency protection", fontsize=9)
    ax.legend(handles=[l1, l2], fontsize=7, loc="center left")

    ax = axes[1]
    ax.plot(LABELS, whale_ttft_mean, "o-", color="#c44e52", label="whale TTFT mean")
    ax.plot(LABELS, whale_ttft_p99, "s--", color="#dd8452", label="whale TTFT p99")
    ax.set_ylabel("whale TTFT (ms)")
    ax.set_title("(b) ...at a whale-TTFT cost\n(non-monotonic: 2048 is best)", fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle("Ramp-rate benefit vs. latency: the other side of the tradeoff", y=1.03)
    fig.tight_layout()
    fig.savefig("fig_ramp_latency_tradeoff.png", dpi=200, bbox_inches="tight")
    print("wrote fig_ramp_latency_tradeoff.png")
    print("whale_ttft_mean:", list(zip(BUDGETS, [round(v) for v in whale_ttft_mean])))
    print("whale_ttft_p99:", list(zip(BUDGETS, [round(v) for v in whale_ttft_p99])))
    print("short_tbt_p99:", list(zip(BUDGETS, [round(v, 1) for v in short_tbt_p99])))


if __name__ == "__main__":
    main()
