#!/usr/bin/env python3
"""Analyze Phase 1.3: near-saturation headline experiment.

Compares ns_base_{dataset} (baseline) vs ns_comb_{dataset} (combined) for
BurstGPT and ShareGPT at near-saturation arrival rate.  Includes KV cache hit
rate if it was recorded by run_near_sat_bench.sh.

Usage:
    python3 src/analyze_near_sat.py [--log-dir logs]
"""
import argparse
import glob
import json
import os
import sys

DATASETS = ["burstgpt", "sharegpt"]

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
    return f"{pct:+.1f}% {arrow}"


def report_dataset(ds, logs):
    base_tag  = f"ns_base_{ds}"
    comb_tag  = f"ns_comb_{ds}"
    aging_tag = f"ns_aging_{ds}"
    base  = logs.get(base_tag)
    comb  = logs.get(comb_tag)
    aging = logs.get(aging_tag)

    print(f"\n{'='*80}")
    print(f"  {ds.upper()}  —  combined / aging vs baseline at near-saturation")
    print(f"{'='*80}")

    for tag, name in [(base_tag, "baseline"), (comb_tag, "combined"), (aging_tag, "aging")]:
        if not logs.get(tag):
            print(f"  [WARNING] Missing {tag} — run scripts/run_near_sat_bench.sh")

    if not base:
        return

    conditions = [("baseline", base), ("combined", comb), ("aging (T=2s)", aging)]

    col = 16
    hdr = f"  {'Metric':<24}"
    for name, _ in conditions:
        hdr += f"  {name:>{col}}"
    print(hdr)
    print("  " + "-" * (24 + len(conditions) * (col + 2)))

    for label, key, lower_better in METRICS:
        is_pct = key == "kv_hit_rate"
        is_tp = "Throughput" in label
        p = 2 if is_tp else 1
        bv = base.get(key)
        row = f"  {label:<24}"
        for i, (name, d) in enumerate(conditions):
            v = d.get(key) if d else None
            if i == 0:
                cell = fmt(bv, p, pct=is_pct)
            else:
                if is_pct:
                    cell = (f"{(v - bv)*100:+.1f}pp"
                            if bv is not None and v is not None else "N/A")
                else:
                    cell = delta_str(bv, v, lower_better)
            row += f"  {cell:>{col}}"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    logs = load_logs(args.log_dir)
    if not logs:
        print(f"No JSON results in {args.log_dir}")
        sys.exit(1)

    print("\nPhase 1.3 — Near-Saturation Headline Experiment")
    print(f"Log dir : {args.log_dir}")
    print("Tags    : ns_base_{{dataset}} / ns_comb_{{dataset}}")
    print("Arrow   : <-- = combined improves  >>> = combined degrades")

    for ds in DATASETS:
        report_dataset(ds, logs)

    print()


if __name__ == "__main__":
    main()
