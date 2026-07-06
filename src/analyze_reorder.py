#!/usr/bin/env python3
"""Analyze prefix-aware reordering experiment results.

Usage:
    python3 src/analyze_reorder.py [--log-dir logs]
"""
import argparse
import glob
import json
import os
import sys


def load_logs(log_dir):
    data = {}
    for f in glob.glob(os.path.join(log_dir, "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = d.get("tag") or os.path.splitext(os.path.basename(f))[0]
        if tag not in data:
            data[tag] = d
    return data


def fmt(v, p=1):
    return "{:8.{}f}".format(v, p) if v is not None else "     N/A"


def delta(base, val):
    if base is None or val is None or base == 0:
        return "       "
    return "{:+6.1f}%".format((val - base) / base * 100)


def report(dataset, off, on):
    print("\n" + "=" * 72)
    print("  {}".format(dataset))
    print("=" * 72)
    print("  {:<26}  {:>8}  {:>8}   Delta".format("Metric", "Reorder OFF", "Reorder ON"))
    print("  " + "-" * 66)

    metrics = [
        ("TTFT p50 (ms)",  "p50_ttft_ms",  True),
        ("TTFT p95 (ms)",  "p95_ttft_ms",  True),
        ("TTFT p99 (ms)",  "p99_ttft_ms",  True),
        ("TPOT p50 (ms)",  "p50_tpot_ms",  True),
        ("TPOT p95 (ms)",  "p95_tpot_ms",  True),
        ("E2EL p95 (ms)",  "p95_e2el_ms",  True),
        ("Throughput (req/s)", "request_throughput", False),
    ]
    for label, key, lower_better in metrics:
        bv = off.get(key)
        dv = on.get(key)
        d = delta(bv, dv)
        better = ""
        if bv is not None and dv is not None:
            improved = (dv < bv) if lower_better else (dv > bv)
            better = " <--" if improved else ""
        p = 2 if "Throughput" in label else 1
        print("  {:<26}  {}  {}   {}{}".format(
            label, fmt(bv, p), fmt(dv, p), d, better))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    data = load_logs(args.log_dir)
    if not data:
        print("No JSON results in {}".format(args.log_dir))
        sys.exit(1)

    print("\nPrefix-Aware Request Reordering — results")
    print("Log dir : {}".format(args.log_dir))

    missing = [t for t in ("reorder_off_burstgpt", "reorder_on_burstgpt",
                            "reorder_off_sharegpt", "reorder_on_sharegpt")
               if t not in data]
    if missing:
        print("Missing tags: {}".format(missing))

    for dataset, off_key, on_key in [
        ("BurstGPT", "reorder_off_burstgpt", "reorder_on_burstgpt"),
        ("ShareGPT", "reorder_off_sharegpt", "reorder_on_sharegpt"),
    ]:
        if off_key in data and on_key in data:
            report(dataset, data[off_key], data[on_key])
        else:
            print("\n[WARN] Skipping {} — data not yet available".format(dataset))

    print()


if __name__ == "__main__":
    main()
