#!/usr/bin/env python3
"""Analyze aging mechanism experiment results.

Compares base_r8_burstgpt, comb_r8_burstgpt (from rate sweep), and any
aging_r8_burstgpt tags (from run_aging_bench.sh) found in the log dir.

Usage:
    python3 src/analyze_aging.py [--log-dir logs]

Multiple aging runs with different thresholds will each appear if they were
saved with distinct tags. To tag a run, set RESULT_TAG before running:
    RESULT_TAG=aging_2s_r8_burstgpt AGING_THRESHOLD_MS=2000 bash scripts/run_aging_bench.sh
"""
import argparse
import glob
import json
import os
import sys

METRICS = [
    ("TTFT p50 (ms)",       "p50_ttft_ms",        True),
    ("TTFT p95 (ms)",       "p95_ttft_ms",        True),
    ("TPOT p50 (ms)",       "p50_tpot_ms",        True),
    ("TPOT p95 (ms)",       "p95_tpot_ms",        True),
    ("E2EL p95 (ms)",       "p95_e2el_ms",        True),
    ("Throughput (req/s)",  "request_throughput",  False),
]

REFERENCE_TAGS = ["base_r8_burstgpt", "comb_r8_burstgpt"]


def load_logs(log_dir):
    data = {}
    for f in sorted(glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        tag = d.get("tag") or os.path.splitext(os.path.basename(f))[0]
        data[tag] = d  # later mtime wins
    return data


def delta_str(base_val, val, lower_better):
    if base_val is None or val is None or base_val == 0:
        return "     N/A"
    pct = (val - base_val) / base_val * 100
    arrow = ("<--" if val < base_val else ">>>") if lower_better else ("<--" if val > base_val else ">>>")
    return "%+6.1f%% %s" % (pct, arrow)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    logs = load_logs(args.log_dir)
    if not logs:
        print("No JSON results in", args.log_dir)
        sys.exit(1)

    # reference conditions
    base = logs.get("base_r8_burstgpt")
    comb = logs.get("comb_r8_burstgpt")

    # collect aging conditions: any tag containing "aging" and "burstgpt"
    aging_tags = sorted(
        t for t in logs if "aging" in t and "burstgpt" in t
    )

    if not base:
        print("WARNING: base_r8_burstgpt not found — run scripts/run_rate_sweep.sh first")
    if not comb:
        print("WARNING: comb_r8_burstgpt not found — run scripts/run_rate_sweep.sh first")
    if not aging_tags:
        print("WARNING: no aging_*_burstgpt tags found — run scripts/run_aging_bench.sh first")

    all_conditions = (
        [("baseline", base), ("comb (T=inf)", comb)]
        + [(t, logs[t]) for t in aging_tags]
    )

    # header
    col_w = 14
    header = "  %-22s" % "Metric"
    for label, _ in all_conditions:
        header += "  %*s" % (col_w, label)
    print("\nAging Mechanism — BurstGPT 8 req/s")
    print("Log dir:", args.log_dir)
    print("Arrow   : <-- = improvement vs baseline  >>> = regression")
    print()
    print(header)
    print("  " + "-" * (24 + (col_w + 2) * len(all_conditions)))

    for label, key, lower_better in METRICS:
        row = "  %-22s" % label
        bv = base.get(key) if base else None
        for cond_label, d in all_conditions:
            v = d.get(key) if d else None
            if cond_label == "baseline":
                cell = "%*.1f" % (col_w, v) if v is not None else "%*s" % (col_w, "N/A")
            else:
                cell = "%*s" % (col_w, delta_str(bv, v, lower_better))
            row += "  " + cell
        print(row)

    # absolute values for all conditions
    print()
    print("  Absolute values:")
    abs_header = "  %-22s" % "Metric"
    for label, _ in all_conditions:
        abs_header += "  %*s" % (col_w, label)
    print(abs_header)
    print("  " + "-" * (24 + (col_w + 2) * len(all_conditions)))
    for label, key, _ in METRICS:
        p = 2 if "Throughput" in label else 1
        row = "  %-22s" % label
        for _, d in all_conditions:
            v = d.get(key) if d else None
            cell = "%*.{}f".format(p) % (col_w, v) if v is not None else "%*s" % (col_w, "N/A")
            row += "  " + cell
        print(row)

    print()


if __name__ == "__main__":
    main()
