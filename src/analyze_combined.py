#!/usr/bin/env python3
"""Analyze the 4-condition combined experiment.

Conditions:
  1. baseline      — reorder OFF, dynamic chunk OFF (static 2048)
  2. dynamic only  — reorder OFF, dynamic chunk ON
  3. reorder only  — reorder ON,  dynamic chunk OFF
  4. combined      — reorder ON,  dynamic chunk ON  ← new

Usage:
    python3 src/analyze_combined.py [--log-dir logs]
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
    return "{:9.{}f}".format(v, p) if v is not None else "      N/A"


def delta(base, val):
    if base is None or val is None or base == 0:
        return "       "
    return "{:+6.1f}%".format((val - base) / base * 100)


METRICS = [
    ("TTFT p50 (ms)",       "p50_ttft_ms",          True),
    ("TTFT p95 (ms)",       "p95_ttft_ms",          True),
    ("TPOT p50 (ms)",       "p50_tpot_ms",          True),
    ("TPOT p95 (ms)",       "p95_tpot_ms",          True),
    ("E2EL p95 (ms)",       "p95_e2el_ms",          True),
    ("Throughput (req/s)",  "request_throughput",   False),
]

CONDITIONS = [
    ("Baseline",      "baseline"),
    ("Dynamic only",  "dynamic"),
    ("Reorder only",  "reorder_on"),
    ("Combined",      "combined"),
]


def report(dataset, logs):
    suffixes = {label: tag for label, tag in CONDITIONS}
    keys = {label: "{}_{}".format(tag, dataset.lower()) for label, tag in CONDITIONS}

    print("\n" + "=" * 90)
    print("  {}".format(dataset))
    print("=" * 90)
    header = "  {:<22}".format("Metric")
    for label, _ in CONDITIONS:
        header += "  {:>9}".format(label[:9])
    header += "   vs Baseline"
    print(header)
    print("  " + "-" * 86)

    base_key = keys["Baseline"]
    base = logs.get(base_key, {})

    for label, key, lower_better in METRICS:
        p = 2 if "Throughput" in label else 1
        row = "  {:<22}".format(label)
        vals = {}
        for cond_label, _ in CONDITIONS:
            v = logs.get(keys[cond_label], {}).get(key)
            vals[cond_label] = v
            row += "  {}".format(fmt(v, p))

        # delta for combined vs baseline
        bv = vals["Baseline"]
        cv = vals["Combined"]
        d = delta(bv, cv)
        better = ""
        if bv is not None and cv is not None:
            improved = (cv < bv) if lower_better else (cv > bv)
            better = " <--" if improved else ""
        row += "   {}{}".format(d, better)
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    data = load_logs(args.log_dir)
    if not data:
        print("No JSON results in {}".format(args.log_dir))
        sys.exit(1)

    print("\nCombined Condition Experiment — 4-way comparison")
    print("Log dir : {}".format(args.log_dir))
    print("Conditions: baseline | dynamic only | reorder only | combined")

    expected = [
        "baseline_burstgpt", "baseline_sharegpt",
        "dynamic_burstgpt",  "dynamic_sharegpt",
        "reorder_on_burstgpt", "reorder_on_sharegpt",
        "combined_burstgpt", "combined_sharegpt",
    ]
    missing = [t for t in expected if t not in data]
    if missing:
        print("Missing tags: {}".format(missing))

    for dataset in ["BurstGPT", "ShareGPT"]:
        report(dataset, data)

    print()


if __name__ == "__main__":
    main()
