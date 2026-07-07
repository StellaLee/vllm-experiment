#!/usr/bin/env python3
"""Analyze Ablation B: 2x2 LRU factorial (reordering x chunk control) at near-saturation.

Loads all 4 LRU conditions from run_near_sat_bench.sh and run_lru_factorial.sh.
CF eviction conditions are excluded (policy flaw; deferred to Phase 2).

All 4 tags:
  ns_base_{ds}     LRU | reorder=off | chunk=static   (from run_near_sat_bench.sh)
  ns_dyn_{ds}      LRU | reorder=off | chunk=dynamic  (from run_lru_factorial.sh)
  ns_reorder_{ds}  LRU | reorder=on  | chunk=static   (from run_lru_factorial.sh)
  ns_comb_{ds}     LRU | reorder=on  | chunk=dynamic  (from run_near_sat_bench.sh)

Usage:
    python3 src/analyze_factorial.py [--log-dir logs] [--dataset burstgpt|sharegpt]
"""
import argparse
import glob
import json
import os
import sys

DATASETS = ["burstgpt", "sharegpt"]

# (label, tag_prefix)
CONDITIONS = [
    ("off|static",  "ns_base"),
    ("off|dynamic", "ns_dyn"),
    ("on|static",   "ns_reorder"),
    ("on|dynamic",  "ns_comb"),
]

METRICS = [
    ("TTFT p50 (ms)",      "p50_ttft_ms",        True),
    ("TTFT p95 (ms)",      "p95_ttft_ms",        True),
    ("TPOT p50 (ms)",      "p50_tpot_ms",        True),
    ("TPOT p95 (ms)",      "p95_tpot_ms",        True),
    ("E2EL p95 (ms)",      "p95_e2el_ms",        True),
    ("Throughput (req/s)", "request_throughput",  False),
    ("KV hit rate",        "kv_hit_rate",         False),
]


def load_logs(log_dir):
    data = {}
    for f in sorted(glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = d.get("tag") or os.path.splitext(os.path.basename(f))[0]
        data[tag] = d
    return data


def fmt(v, p=1, pct=False):
    if v is None:
        return "N/A"
    if pct:
        return f"{v:.1%}"
    return f"{v:.{p}f}"


def delta_str(bv, v, lower_better):
    if bv is None or v is None or bv == 0:
        return "N/A"
    pct = (v - bv) / abs(bv) * 100
    arrow = ("<--" if v < bv else ">>>") if lower_better else ("<--" if v > bv else ">>>")
    return f"{pct:+.1f}%{arrow}"


def report_dataset(ds, logs):
    print(f"\n{'='*80}")
    print(f"  {ds.upper()}  —  2x2 LRU factorial at near-saturation  (delta vs off|static baseline)")
    print(f"  Columns: reorder x chunk_control")
    print(f"{'='*80}")

    base_tag = f"ns_base_{ds}"
    base = logs.get(base_tag)
    if not base:
        print(f"  [WARNING] Baseline {base_tag} not found — run run_near_sat_bench.sh first")

    # check which conditions are present
    missing = [
        f"{tag}_{ds}" for _, tag in CONDITIONS
        if f"{tag}_{ds}" not in logs
    ]
    if missing:
        print(f"  [WARNING] Missing tags: {missing}")

    col = 14

    # header row: reorder x chunk matrix
    hdr = f"  {'Metric':<22}"
    for label, _ in CONDITIONS:
        hdr += f"  {label:>{col}}"
    print(hdr)
    print("  " + "-" * (22 + len(CONDITIONS) * (col + 2)))

    for m_label, key, lower_better in METRICS:
        bv = base.get(key) if base else None
        is_pct = key == "kv_hit_rate"
        is_tp = "Throughput" in m_label
        p = 2 if is_tp else 1

        row = f"  {m_label:<22}"
        for i, (cond_label, tag_prefix) in enumerate(CONDITIONS):
            tag = f"{tag_prefix}_{ds}"
            d = logs.get(tag)
            v = d.get(key) if d else None
            if i == 0:
                cell = fmt(v, p, pct=is_pct)
            else:
                if is_pct:
                    if bv is not None and v is not None:
                        cell = f"{(v - bv)*100:+.1f}pp"
                    else:
                        cell = "N/A"
                else:
                    cell = delta_str(bv, v, lower_better)
            row += f"  {cell:>{col}}"
        print(row)

    # ── absolute values ───────────────────────────────────────────────────────
    print()
    print("  Absolute values:")
    abs_hdr = f"  {'Metric':<22}"
    for label, *_ in CONDITIONS:
        abs_hdr += f"  {label:>{col}}"
    print(abs_hdr)
    print("  " + "-" * (22 + len(CONDITIONS) * (col + 2)))
    for m_label, key, _ in METRICS:
        is_pct = key == "kv_hit_rate"
        is_tp = "Throughput" in m_label
        p = 2 if is_tp else 1
        row = f"  {m_label:<22}"
        for _, tag_prefix in CONDITIONS:
            tag = f"{tag_prefix}_{ds}"
            d = logs.get(tag)
            v = d.get(key) if d else None
            row += f"  {fmt(v, p, pct=is_pct):>{col}}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = parser.parse_args()

    logs = load_logs(args.log_dir)
    if not logs:
        print(f"No JSON results in {args.log_dir}")
        sys.exit(1)

    print("\nAblation B — 2x2 LRU Factorial at Near-Saturation")
    print(f"Log dir : {args.log_dir}")
    print("Columns : reorder={off,on} x chunk={static,dynamic}")
    print("Arrow   : <-- = improves vs off|static baseline  >>> = degrades")

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        report_dataset(ds, logs)

    print()


if __name__ == "__main__":
    main()
