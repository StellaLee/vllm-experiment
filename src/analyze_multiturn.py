#!/usr/bin/env python3
"""Analyze multi-turn ShareGPT replay results.

Compares baseline / combined / aging across per-turn TTFT and KV hit rate.
Expects JSONL files written by replay_sharegpt.py and summary JSONs written
by run_multiturn_bench.sh.

Usage:
    python3 src/analyze_multiturn.py [--log-dir logs] [--concurrency 20]
"""
import argparse
import glob
import json
import os
import statistics


def load_jsonl(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_summary(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def per_turn_stats(records, max_turns):
    """Returns {turn: {"n": int, "median_ttft": ms, "p95_ttft": ms}} for turns 1..max_turns."""
    by_turn = {}
    for r in records:
        t = r.get("turn")
        ttft = r.get("ttft")
        if t is None or ttft is None:
            continue
        by_turn.setdefault(t, []).append(ttft * 1000)  # s -> ms

    result = {}
    for turn in range(1, max_turns + 1):
        vals = sorted(by_turn.get(turn, []))
        if not vals:
            result[turn] = {"n": 0, "median": None, "p95": None}
            continue
        n = len(vals)
        median = statistics.median(vals)
        p95 = vals[min(n - 1, int(n * 0.95))]
        result[turn] = {"n": n, "median": round(median, 1), "p95": round(p95, 1)}
    return result


def fmt(v, suffix=""):
    if v is None:
        return "N/A"
    return f"{v:.1f}{suffix}"


def pct_delta(base, v):
    if base is None or v is None or base == 0:
        return "N/A"
    d = (v - base) / base * 100
    arrow = "<--" if v < base else ">>>"
    return f"{d:+.1f}% {arrow}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=4)
    args = parser.parse_args()

    ctag = f"c{args.concurrency}"
    conditions = [("baseline", f"mt_base_{ctag}"), ("combined", f"mt_comb_{ctag}"), (f"aging (T=2s)", f"mt_aging_{ctag}")]

    # find the most recent JSONL for each tag
    datasets = {}
    for cond_name, tag in conditions:
        pattern = os.path.join(args.log_dir, f"*-mt-{tag}.jsonl")
        files = sorted(glob.glob(pattern), key=os.path.getmtime)
        if not files:
            print(f"  [WARNING] No JSONL found for {tag} — run scripts/run_multiturn_bench.sh first")
            datasets[cond_name] = ([], {})
            continue
        jsonl = files[-1]
        summary_path = jsonl.replace(".jsonl", "-summary.json")
        records = load_jsonl(jsonl)
        summary = load_summary(summary_path)
        datasets[cond_name] = (records, summary)
        print(f"  Loaded {len(records)} records for {cond_name} from {os.path.basename(jsonl)}")

    print(f"\nMulti-Turn ShareGPT Replay — concurrency={args.concurrency}  max_turns={args.max_turns}")
    print("Arrow: <-- = improvement over baseline  >>> = regression\n")

    # Per-turn TTFT table
    turn_stats = {name: per_turn_stats(recs, args.max_turns)
                  for name, (recs, _) in datasets.items()}

    base_name = "baseline"
    col = 20
    print(f"{'Turn':<6}{'n':>5}  {'Baseline med':>{col}}  {'Baseline p95':>{col}}", end="")
    for cond_name, _ in conditions[1:]:
        print(f"  {cond_name+' med':>{col}}  {cond_name+' p95':>{col}}", end="")
    print()
    print("-" * (6 + 5 + 2 + (col + 2) * 2 + (col + 2) * 2 * (len(conditions) - 1)))

    for turn in range(1, args.max_turns + 1):
        base_s = turn_stats[base_name].get(turn, {})
        n = base_s.get("n", 0)
        b_med = base_s.get("median")
        b_p95 = base_s.get("p95")
        row = f"Turn {turn:<1}{n:>5}  {fmt(b_med, 'ms'):>{col}}  {fmt(b_p95, 'ms'):>{col}}"
        for cond_name, _ in conditions[1:]:
            s = turn_stats[cond_name].get(turn, {})
            row += f"  {fmt(s.get('median'), 'ms'):>{col}}  {fmt(s.get('p95'), 'ms'):>{col}}"
        print(row)

    # Delta table
    print(f"\n{'Turn':<6}{'n':>5}", end="")
    for cond_name, _ in conditions[1:]:
        print(f"  {cond_name+' Δmed':>{col}}  {cond_name+' Δp95':>{col}}", end="")
    print()
    print("-" * (6 + 5 + (col + 2) * 2 * (len(conditions) - 1)))

    for turn in range(1, args.max_turns + 1):
        base_s = turn_stats[base_name].get(turn, {})
        n = base_s.get("n", 0)
        b_med = base_s.get("median")
        b_p95 = base_s.get("p95")
        row = f"Turn {turn:<1}{n:>5}"
        for cond_name, _ in conditions[1:]:
            s = turn_stats[cond_name].get(turn, {})
            row += f"  {pct_delta(b_med, s.get('median')):>{col}}  {pct_delta(b_p95, s.get('p95')):>{col}}"
        print(row)

    # KV hit rate summary
    print("\nKV Cache Hit Rate:")
    for cond_name, (_, summary) in datasets.items():
        kv = summary.get("kv_hit_rate")
        hits = summary.get("kv_hits", 0)
        queries = summary.get("kv_queries", 0)
        if kv is not None:
            print(f"  {cond_name:<20}  {kv:.1%}  ({hits:.0f}/{queries:.0f} blocks)")
        else:
            print(f"  {cond_name:<20}  N/A")

    # Request count sanity check
    print("\nRequest counts per turn:")
    for cond_name, (recs, _) in datasets.items():
        by_turn = {}
        for r in recs:
            by_turn[r.get("turn")] = by_turn.get(r.get("turn"), 0) + 1
        counts = [by_turn.get(t, 0) for t in range(1, args.max_turns + 1)]
        print(f"  {cond_name:<20}  {counts}")


if __name__ == "__main__":
    main()
