#!/usr/bin/env python3
"""Analyze Ablation A: arrival rate sweep.

Loads base_r{rate}_{dataset} and comb_r{rate}_{dataset} tags from the log dir,
prints a table of delta% for each metric at each rate.

Usage:
    python3 src/analyze_rate_sweep.py [--log-dir logs]
"""
import argparse
import glob
import json
import os
import sys

RATES = [1, 2, 4, 8]
DATASETS = ["burstgpt", "sharegpt"]

METRICS = [
    ("TTFT p50 (ms)",      "p50_ttft_ms",        True),
    ("TTFT p95 (ms)",      "p95_ttft_ms",        True),
    ("TPOT p50 (ms)",      "p50_tpot_ms",        True),
    ("TPOT p95 (ms)",      "p95_tpot_ms",        True),
    ("E2EL p95 (ms)",      "p95_e2el_ms",        True),
    ("Throughput (req/s)", "request_throughput",  False),
]


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


def delta_str(base, val, lower_better):
    if base is None or val is None or base == 0:
        return "    N/A "
    pct = (val - base) / base * 100
    arrow = ""
    if lower_better:
        arrow = " <--" if val < base else " >>>"
    else:
        arrow = " <--" if val > base else " >>>"
    return "{:+6.1f}%{}".format(pct, arrow)


def report_dataset(dataset, logs):
    print("\n" + "=" * 88)
    print("  {}  —  combined vs baseline delta by arrival rate".format(dataset.upper()))
    print("=" * 88)

    # header
    header = "  {:<22}".format("Metric")
    for r in RATES:
        header += "  {:>14}".format("{} req/s".format(r))
    print(header)
    print("  " + "-" * 84)

    missing = []
    for r in RATES:
        bk = "base_r{}_{}".format(r, dataset)
        ck = "comb_r{}_{}".format(r, dataset)
        if bk not in logs:
            missing.append(bk)
        if ck not in logs:
            missing.append(ck)

    if missing:
        print("  [WARNING] Missing tags: {}".format(missing))

    for label, key, lower_better in METRICS:
        p = 2 if "Throughput" in label else 1
        row = "  {:<22}".format(label)
        for r in RATES:
            bk = "base_r{}_{}".format(r, dataset)
            ck = "comb_r{}_{}".format(r, dataset)
            bv = logs.get(bk, {}).get(key)
            cv = logs.get(ck, {}).get(key)
            row += "  {:>14}".format(delta_str(bv, cv, lower_better).strip())
        print(row)

    # also print absolute values for baseline at each rate (sanity check)
    print()
    print("  Baseline absolute values:")
    for label, key, _ in METRICS:
        p = 2 if "Throughput" in label else 1
        row = "  {:<22}".format(label)
        for r in RATES:
            bk = "base_r{}_{}".format(r, dataset)
            v = logs.get(bk, {}).get(key)
            row += "  {}".format(fmt(v, p))
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    logs = load_logs(args.log_dir)
    if not logs:
        print("No JSON results in {}".format(args.log_dir))
        sys.exit(1)

    print("\nAblation A — Arrival Rate Sweep")
    print("Log dir : {}".format(args.log_dir))
    print("Rates   : {} req/s".format(RATES))
    print("Tags    : base_r{{rate}}_{{dataset}} / comb_r{{rate}}_{{dataset}}")
    print("Arrow   : <-- = combined improves  >>> = combined degrades")

    for ds in DATASETS:
        report_dataset(ds, logs)

    print()


if __name__ == "__main__":
    main()
