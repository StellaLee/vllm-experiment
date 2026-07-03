#!/usr/bin/env python3
"""
compare_eviction.py — Compare per-turn TTFT and TPOT across eviction policies.

Reads JSONL files produced by replay_sharegpt.py (one per policy) and emits
a markdown findings report with per-turn TTFT + TPOT tables.
"""
import argparse
import json
import math
import os
import statistics
from datetime import date


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def percentile(data, p):
    if not data:
        return float("nan")
    data = sorted(data)
    idx = (len(data) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] * (1 - (idx - lo)) + data[hi] * (idx - lo)


def per_turn_stats(records):
    by_turn = {}
    for r in records:
        t = r.get("turn")
        if t is None:
            continue
        by_turn.setdefault(t, {"ttft": [], "tpot": []})
        if r.get("ttft") is not None:
            by_turn[t]["ttft"].append(r["ttft"])
        if r.get("tpot") is not None:
            by_turn[t]["tpot"].append(r["tpot"])
    result = {}
    for t, vals in sorted(by_turn.items()):
        ttft = vals["ttft"]
        tpot = vals["tpot"]
        result[t] = {
            "n": len(ttft),
            "ttft_median": statistics.median(ttft) if ttft else float("nan"),
            "ttft_p95":    percentile(ttft, 95),
            "tpot_median": statistics.median(tpot) * 1000 if tpot else float("nan"),
            "tpot_p95":    percentile(tpot, 95) * 1000 if tpot else float("nan"),
        }
    return result


def fms(v):
    return "N/A" if math.isnan(v) else f"{v*1000:.1f}ms"

def fms_raw(v):
    return "N/A" if math.isnan(v) else f"{v:.1f}ms"


def pct_str(baseline, val):
    if math.isnan(baseline) or math.isnan(val) or baseline == 0:
        return "N/A"
    p = (baseline - val) / baseline * 100
    return f"{p:+.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lru",    required=True)
    ap.add_argument("--tdf",    default=None)
    ap.add_argument("--cf",     required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lambda-val",  type=float, default=0.1)
    ap.add_argument("--concurrency", type=int,   default=20)
    ap.add_argument("--num-convs",   type=int,   default=200)
    ap.add_argument("--model",  default="Qwen2.5-Coder-7B-Instruct")
    args = ap.parse_args()

    lru_s = per_turn_stats(load_jsonl(args.lru))
    cf_s  = per_turn_stats(load_jsonl(args.cf))
    tdf_s = per_turn_stats(load_jsonl(args.tdf)) if args.tdf else {}

    all_turns = sorted(set(list(lru_s) + list(cf_s) + list(tdf_s)))

    lines = []
    lines.append(f"# Eviction Policy Comparison — {date.today()}")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Model | {args.model} |")
    lines.append("| gpu-memory-utilization | 0.7 |")
    lines.append(f"| Concurrency | {args.concurrency} |")
    lines.append(f"| Conversations | {args.num_convs} (all ≥4 turns) |")
    lines.append("| Dataset | ShareGPT v3 |")
    lines.append("| Max turns | 4 |")
    lines.append("| Policies | LRU (baseline), TDF λ=%.1f, CF |" % args.lambda_val)
    lines.append("")
    lines.append("| Policy | Score formula |")
    lines.append("|--------|---------------|")
    lines.append("| LRU | least-recently-used (vLLM default) |")
    lines.append("| TDF | `(hit_count+1)·exp(−λ·age)` |")
    lines.append("| **CF** | `(hit_count+1)/(prefix_depth+1)` |")
    lines.append("")

    # ── TTFT table ────────────────────────────────────────────────────────
    lines.append("## TTFT (Time to First Token)")
    lines.append("")
    header = "| Turn | n | LRU med | LRU P95 |"
    sep    = "|------|---|---------|---------|"
    if tdf_s:
        header += " TDF P95 |"
        sep    += "---------|"
    header += " CF med | CF P95 | CF vs LRU P95 |"
    sep    += "--------|--------|---------------|"
    lines.append(header)
    lines.append(sep)

    for t in all_turns:
        l = lru_s.get(t, {}); c = cf_s.get(t, {}); td = tdf_s.get(t, {})
        lp = l.get("ttft_p95", float("nan"))
        cp = c.get("ttft_p95", float("nan"))
        row = f"| {t} | {l.get('n',0)} | {fms(l.get('ttft_median', float('nan')))} | {fms(lp)} |"
        if tdf_s:
            row += f" {fms(td.get('ttft_p95', float('nan')))} |"
        row += f" {fms(c.get('ttft_median', float('nan')))} | {fms(cp)} | {pct_str(lp, cp)} |"
        lines.append(row)

    lines.append("")

    # ── TPOT table ────────────────────────────────────────────────────────
    lines.append("## TPOT (Time per Output Token, decode phase only)")
    lines.append("")
    lines.append("TPOT = (latency − TTFT) / output_words. Output words used as token proxy (~0.75 words/token).")
    lines.append("Lower TPOT = higher decode throughput.")
    lines.append("")
    header2 = "| Turn | LRU med | LRU P95 |"
    sep2    = "|------|---------|---------|"
    if tdf_s:
        header2 += " TDF P95 |"
        sep2    += "---------|"
    header2 += " CF med | CF P95 | CF vs LRU P95 |"
    sep2    += "--------|--------|---------------|"
    lines.append(header2)
    lines.append(sep2)

    for t in all_turns:
        l = lru_s.get(t, {}); c = cf_s.get(t, {}); td = tdf_s.get(t, {})
        lp = l.get("tpot_p95", float("nan"))
        cp = c.get("tpot_p95", float("nan"))
        row = f"| {t} | {fms_raw(l.get('tpot_median', float('nan')))} | {fms_raw(lp)} |"
        if tdf_s:
            row += f" {fms_raw(td.get('tpot_p95', float('nan')))} |"
        row += f" {fms_raw(c.get('tpot_median', float('nan')))} | {fms_raw(cp)} | {pct_str(lp, cp)} |"
        lines.append(row)

    lines.append("")

    # ── Summary ───────────────────────────────────────────────────────────
    lines.append("## Turn-4 Summary")
    lines.append("")
    lines.append("| Metric | LRU | CF | Improvement |")
    lines.append("|--------|-----|----|-------------|")
    for metric, label, scale in [
        ("ttft_p95",  "P95 TTFT",  1000),
        ("ttft_median","Median TTFT",1000),
        ("tpot_p95",  "P95 TPOT",  1),
        ("tpot_median","Median TPOT",1),
    ]:
        l4 = lru_s.get(4, {}).get(metric, float("nan")) * scale
        c4 = cf_s.get(4,  {}).get(metric, float("nan")) * scale
        imp = pct_str(l4 / scale, c4 / scale) if scale == 1000 else pct_str(l4, c4)
        lines.append(f"| {label} | {l4:.1f}ms | {c4:.1f}ms | {imp} |")

    lines.append("")
    lines.append("## Raw counts")
    lines.append("")
    lines.append("| Turn | LRU | CF |")
    lines.append("|------|-----|-----|")
    for t in all_turns:
        lines.append(f"| {t} | {lru_s.get(t,{}).get('n',0)} | {cf_s.get(t,{}).get('n',0)} |")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Findings written to {args.output}")

    print("\n=== Turn-by-turn summary ===")
    print(f"{'Turn':>4}  {'n':>4}  {'LRU P95 TTFT':>14}  {'CF P95 TTFT':>13}  {'Δ TTFT':>8}  {'LRU med TPOT':>14}  {'CF med TPOT':>13}")
    for t in all_turns:
        l = lru_s.get(t, {}); c = cf_s.get(t, {})
        lp = l.get("ttft_p95", float("nan")) * 1000
        cp = c.get("ttft_p95", float("nan")) * 1000
        lt = l.get("tpot_median", float("nan"))
        ct = c.get("tpot_median", float("nan"))
        print(f"{t:>4}  {l.get('n',0):>4}  {lp:>13.1f}ms  {cp:>12.1f}ms  {pct_str(lp/1000,cp/1000):>8}  {lt:>13.1f}ms  {ct:>12.1f}ms")


if __name__ == "__main__":
    main()
