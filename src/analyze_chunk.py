#!/usr/bin/env python3
"""Analyse dynamic chunk size experiment results.

Usage:
    python3 analyze_chunk.py <results_dir>

Reads JSON files produced by `vllm bench serve --save-result` and prints a
side-by-side comparison of baseline vs. dynamic chunk size metrics.
"""
import json
import glob
import os
import sys


def _fmt_ms(v):
    return "{:9.1f}".format(v) if v is not None else "      N/A"


def _fmt_rps(v):
    return "{:9.2f}".format(v) if v is not None else "      N/A"


def _delta(bv, dv):
    if bv is None or dv is None or bv == 0:
        return ""
    pct = (dv - bv) / bv * 100
    sign = "-" if pct < 0 else "+"
    return "  {:s}{:.1f}%".format(sign, abs(pct))


def print_row(label, b, d, key, fmt=_fmt_ms):
    bv = b.get(key)
    dv = d.get(key)
    print("  {:<26} {}   {}{}".format(label, fmt(bv), fmt(dv), _delta(bv, dv)))


def report(dataset_name, b, d):
    print("\n" + "=" * 66)
    print("  {}  (n={} prompts, request_rate=inf)".format(
        dataset_name, b.get("num_prompts", "?")))
    print("=" * 66)
    print("  {:<26} {:>9}   {:>9}   Delta".format(
        "Metric", "Baseline", "Dynamic"))
    print("  " + "-" * 60)

    for p in ("p50", "p95", "p99"):
        print_row("TTFT {} (ms)".format(p), b, d, "{}_ttft_ms".format(p))
    print()
    for p in ("p50", "p95", "p99"):
        print_row("TPOT {} (ms)".format(p), b, d, "{}_tpot_ms".format(p))
    print()
    for p in ("p50", "p95", "p99"):
        print_row("E2EL {} (ms)".format(p), b, d, "{}_e2el_ms".format(p))
    print()
    print_row("Mean TTFT (ms)", b, d, "mean_ttft_ms")
    print_row("Throughput (req/s)", b, d, "request_throughput", _fmt_rps)
    print_row("Output tok/s", b, d, "output_throughput", _fmt_rps)
    print("  {:<26} {:>9}   {:>9}".format(
        "Total output tokens",
        str(b.get("total_output_tokens", "?")),
        str(d.get("total_output_tokens", "?"))))
    print("  {:<26} {:>9}   {:>9}".format(
        "Completed",
        str(b.get("completed", "?")),
        str(d.get("completed", "?"))))


def main():
    if len(sys.argv) < 2:
        print("Usage: analyze_chunk.py <results_dir>")
        sys.exit(1)

    results_dir = sys.argv[1]
    data = {}
    for f in glob.glob(os.path.join(results_dir, "*.json")):
        d = json.load(open(f))
        tag = d.get("tag") or os.path.splitext(os.path.basename(f))[0]
        data[tag] = d

    if not data:
        print("No JSON results found in {}".format(results_dir))
        sys.exit(1)

    print("\nChunk Size Experiment")
    print("Results dir : {}".format(results_dir))
    print("Runs found  : {}".format(sorted(data.keys())))
    print("Baseline    : static max_num_scheduled_tokens (DYNAMIC_CHUNK=0)")
    print("Dynamic     : bang-bang controller (DYNAMIC_CHUNK=1, target=8 decoders)")

    for dataset in ("burstgpt", "sharegpt"):
        bk = "baseline_" + dataset
        dk = "dynamic_" + dataset
        if bk in data and dk in data:
            report(dataset.upper(), data[bk], data[dk])
        else:
            missing = [k for k in (bk, dk) if k not in data]
            print("\n[WARN] Missing runs for {}: {}".format(dataset, missing))


if __name__ == "__main__":
    main()
