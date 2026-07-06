#!/usr/bin/env python3
"""Analyze static chunk size sweep and compare with dynamic controller.

Loads all relevant JSONs from logs/ and prints a tradeoff table showing
TTFT vs TPOT across chunk sizes, with the dynamic controller as the final row.

Usage:
    python3 src/analyze_chunk_sweep.py [--log-dir logs]
"""
import argparse
import glob
import json
import os
import sys

SWEEP_SIZES = [256, 1024, 2048, 4096]


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


def fmt_ms(v):
    return "{:8.1f}".format(v) if v is not None else "     N/A"


def fmt_rps(v):
    return "{:8.2f}".format(v) if v is not None else "     N/A"


def pct_delta(base, val):
    if base is None or val is None or base == 0:
        return "       "
    p = (val - base) / base * 100
    return "{:+6.1f}%".format(p)


def report_dataset(dataset_label, rows_with_labels, baseline_idx):
    print("\n" + "=" * 90)
    print("  {}".format(dataset_label))
    print("=" * 90)
    hdr = "  {:<24}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}"
    print(hdr.format("Config", "TTFT p50", "TTFT p95", "TPOT p50", "TPOT p95",
                     "E2EL p95", "Req/s"))
    print("  " + "-" * 84)

    brow = rows_with_labels[baseline_idx][1]

    for label, row in rows_with_labels:
        t50  = row.get("p50_ttft_ms")
        t95  = row.get("p95_ttft_ms")
        tp50 = row.get("p50_tpot_ms")
        tp95 = row.get("p95_tpot_ms")
        e95  = row.get("p95_e2el_ms")
        rps  = row.get("request_throughput")

        marker = " *" if "dynamic" in label else "  "
        print("  {:<24}{}  {}  {}  {}  {}  {}  {}".format(
            label, marker,
            fmt_ms(t50), fmt_ms(t95),
            fmt_ms(tp50), fmt_ms(tp95),
            fmt_ms(e95), fmt_rps(rps)))

    # Delta table vs 2048 baseline
    print()
    print("  Deltas vs static 2048 (default):")
    print("  {:<24}  {:>8}  {:>8}  {:>8}  {:>8}".format(
        "Config", "TTFT p50", "TTFT p95", "TPOT p95", "Req/s"))
    print("  " + "-" * 60)
    for label, row in rows_with_labels:
        if "2048" in label and "dynamic" not in label:
            continue
        d_t50  = pct_delta(brow.get("p50_ttft_ms"),  row.get("p50_ttft_ms"))
        d_t95  = pct_delta(brow.get("p95_ttft_ms"),  row.get("p95_ttft_ms"))
        d_tp95 = pct_delta(brow.get("p95_tpot_ms"),  row.get("p95_tpot_ms"))
        d_rps  = pct_delta(brow.get("request_throughput"), row.get("request_throughput"))
        print("  {:<24}  {}  {}  {}  {}".format(label, d_t50, d_t95, d_tp95, d_rps))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    data = load_logs(args.log_dir)
    if not data:
        print("No JSON results found in {}".format(args.log_dir))
        sys.exit(1)

    print("\nStatic Chunk Size Sweep — tradeoff analysis")
    print("Log dir : {}".format(args.log_dir))
    print("Tags    : {}".format(sorted(data.keys())))

    for dataset, bl_key, dyn_key in [
        ("BurstGPT", "baseline_burstgpt", "dynamic_burstgpt"),
        ("ShareGPT", "baseline_sharegpt", "dynamic_sharegpt"),
    ]:
        ds = dataset.lower()
        rows = []
        baseline_idx = None

        for size in SWEEP_SIZES:
            if size == 2048:
                tag = bl_key
                label = "static 2048 (default)"
            else:
                tag = "sweep_{}_{}".format(size, ds)
                label = "static {}".format(size)

            if tag in data:
                if size == 2048:
                    baseline_idx = len(rows)
                rows.append((label, data[tag]))
            else:
                rows.append((label, {}))

        dyn_label = "dynamic (ctrl)"
        rows.append((dyn_label, data.get(dyn_key, {})))

        if baseline_idx is None:
            print("\n[WARN] No 2048 baseline found for {} — skipping delta table".format(dataset))
            baseline_idx = 0

        report_dataset(dataset, rows, baseline_idx)

    print()


if __name__ == "__main__":
    main()
