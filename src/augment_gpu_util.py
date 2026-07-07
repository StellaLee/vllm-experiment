#!/usr/bin/env python3
"""Augment a benchmark result JSON with average GPU utilization.

Called from run_near_sat_bench.sh after each benchmark run.
Reads the nvidia-smi CSV log captured during the run.

Usage:
    python3 src/augment_gpu_util.py <tag> <log_dir> <nvidia_smi_csv>
"""
import glob
import json
import os
import sys


def main():
    tag, log_dir, csv_path = sys.argv[1:]
    try:
        rows = [l.strip().split(",") for l in open(csv_path) if l.strip()]
        gpu_utils = [float(r[0]) for r in rows if len(r) >= 1 and r[0].strip()]
        mem_utils = [float(r[1]) for r in rows if len(r) >= 2 and r[1].strip()]
        avg_gpu = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0.0
        avg_mem = sum(mem_utils) / len(mem_utils) if mem_utils else 0.0
    except Exception as e:
        print(f"  WARNING: could not parse GPU util from {csv_path}: {e}")
        return

    for f in sorted(
        glob.glob(os.path.join(log_dir, "*.json")), key=os.path.getmtime, reverse=True
    ):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("tag") == tag:
            d["gpu_util_pct"] = avg_gpu
            d["gpu_mem_util_pct"] = avg_mem
            json.dump(d, open(f, "w"), indent=2)
            print(f"  GPU util: {avg_gpu:.1f}%  mem util: {avg_mem:.1f}%  ({len(gpu_utils)} samples)")
            return
    print(f"  WARNING: no JSON found for tag={tag} in {log_dir}")


if __name__ == "__main__":
    main()
