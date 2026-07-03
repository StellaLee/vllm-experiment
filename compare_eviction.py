#!/usr/bin/env python3
"""
compare_eviction.py — Compare per-turn TTFT between LRU and TDF eviction policies.

Reads two JSONL files produced by replay_sharegpt.py and emits a markdown
findings report with a per-turn TTFT table plus summary statistics.
"""
import argparse
import json
import math
import os
import statistics
from datetime import date


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def percentile(data, p):
    if not data:
        return float("nan")
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    frac = idx - lo
    return data[lo] * (1 - frac) + data[hi] * frac


def per_turn_stats(records):
    """Return dict: turn -> {count, median, p95, mean} for non-null TTFTs."""
    by_turn = {}
    for r in records:
        t = r.get("turn")
        ttft = r.get("ttft")
        if t is None or ttft is None:
            continue
        by_turn.setdefault(t, []).append(ttft)
    result = {}
    for t, vals in sorted(by_turn.items()):
        result[t] = {
            "count": len(vals),
            "median": statistics.median(vals),
            "p95": percentile(vals, 95),
            "mean": statistics.mean(vals),
        }
    return result


def fmt(v, unit="ms"):
    if math.isnan(v):
        return "  N/A"
    if unit == "ms":
        return f"{v * 1000:6.1f}"
    return f"{v:6.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lru", required=True)
    ap.add_argument("--tdf", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lambda-val", type=float, default=0.1)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--num-convs", type=int, default=50)
    args = ap.parse_args()

    lru_recs = load_jsonl(args.lru)
    tdf_recs = load_jsonl(args.tdf)

    lru_stats = per_turn_stats(lru_recs)
    tdf_stats = per_turn_stats(tdf_recs)

    all_turns = sorted(set(list(lru_stats) + list(tdf_stats)))

    lines = []
    lines.append(f"# Eviction Policy Comparison — {date.today()}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append("| Model | Qwen2.5-Coder-7B-Instruct |")
    lines.append("| gpu-memory-utilization | 0.7 |")
    lines.append(f"| Concurrency | {args.concurrency} |")
    lines.append(f"| Conversations | {args.num_convs} |")
    lines.append("| Policies | LRU (baseline) vs TDF |")
    lines.append(f"| TDF score | (hit_count+1) × exp(−λ × age), λ={args.lambda_val} |")
    lines.append("")
    lines.append("## Per-Turn TTFT (milliseconds)")
    lines.append("")
    lines.append("| Turn | LRU median | LRU P95 | TDF median | TDF P95 | P95 Δ | P95 improvement |")
    lines.append("|------|-----------|---------|-----------|---------|-------|-----------------|")

    for t in all_turns:
        ls = lru_stats.get(t, {})
        ts = tdf_stats.get(t, {})
        lru_p95 = ls.get("p95", float("nan"))
        tdf_p95 = ts.get("p95", float("nan"))
        lru_med = ls.get("median", float("nan"))
        tdf_med = ts.get("median", float("nan"))
        if not math.isnan(lru_p95) and not math.isnan(tdf_p95):
            delta_ms = (tdf_p95 - lru_p95) * 1000
            pct = (lru_p95 - tdf_p95) / lru_p95 * 100 if lru_p95 > 0 else float("nan")
            delta_str = f"{delta_ms:+.1f} ms"
            pct_str = f"{pct:+.1f}%"
        else:
            delta_str = "N/A"
            pct_str = "N/A"
        lines.append(
            f"| {t} "
            f"| {fmt(lru_med)} ms "
            f"| {fmt(lru_p95)} ms "
            f"| {fmt(tdf_med)} ms "
            f"| {fmt(tdf_p95)} ms "
            f"| {delta_str} "
            f"| {pct_str} |"
        )

    # Summary for turn 4 specifically
    lines.append("")
    lines.append("## Key Finding: Turn 4 TTFT")
    lines.append("")
    if 4 in lru_stats and 4 in tdf_stats:
        lru4 = lru_stats[4]["p95"] * 1000
        tdf4 = tdf_stats[4]["p95"] * 1000
        delta = lru4 - tdf4
        pct = delta / lru4 * 100 if lru4 > 0 else 0
        lines.append(f"- LRU P95 TTFT (turn 4): **{lru4:.1f} ms**")
        lines.append(f"- TDF P95 TTFT (turn 4): **{tdf4:.1f} ms**")
        lines.append(f"- Improvement: **{delta:.1f} ms ({pct:.1f}%)**")
    else:
        lines.append("Turn 4 data not available in one or both runs.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "LRU evicts the least-recently-used block regardless of access frequency. "
        "Under concurrency=20 with gpu-memory-utilization=0.7, evictions are forced, "
        "and turn-4 requests suffer because their earlier-turn prefix blocks have been evicted."
    )
    lines.append("")
    lines.append(
        f"TDF (λ={args.lambda_val}) scores blocks by `(hit_count+1)·exp(−λ·age)`. "
        "Frequently-hit, recently-cached blocks receive higher scores and survive longer. "
        "Turn-4 requests are more likely to find their prefix blocks still in cache, "
        "reducing TTFT by avoiding KV recomputation."
    )
    lines.append("")
    lines.append("## Raw counts")
    lines.append("")
    lines.append("| Turn | LRU requests | TDF requests |")
    lines.append("|------|-------------|-------------|")
    for t in all_turns:
        lc = lru_stats.get(t, {}).get("count", 0)
        tc = tdf_stats.get(t, {}).get("count", 0)
        lines.append(f"| {t} | {lc} | {tc} |")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Findings written to {args.output}")

    # Also print summary to stdout
    print("\n=== Per-Turn TTFT Summary ===")
    print(f"{'Turn':<6} {'LRU P95':>10} {'TDF P95':>10} {'Improvement':>14}")
    for t in all_turns:
        ls = lru_stats.get(t, {})
        ts = tdf_stats.get(t, {})
        lp = ls.get("p95", float("nan")) * 1000
        tp = ts.get("p95", float("nan")) * 1000
        if not math.isnan(lp) and not math.isnan(tp):
            imp = f"{(lp-tp)/lp*100:+.1f}%" if lp > 0 else "N/A"
        else:
            imp = "N/A"
        print(f"{t:<6} {lp:>10.1f} ms {tp:>10.1f} ms {imp:>14}")


if __name__ == "__main__":
    main()
