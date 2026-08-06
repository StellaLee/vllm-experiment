#!/usr/bin/env python3
"""Standalone NVML power/energy side-car logger.

Runs independently of the replay harness -- correlates to request records
after the fact via wall-clock `time.time()` (matches replay_sharegpt.py's
per-record `ts`/`latency` fields). Samples power (instantaneous, W) and the
hardware cumulative energy counter (mJ, monotonic since driver load) at a
fixed interval for one or more GPU indices. Also logs temperature so a
thermal-warmup check can confirm clock/thermal steady-state before a
measured run starts.

Usage:
  power_logger.py --gpus 0 --interval-ms 50 --output logs/power_trace.csv
  power_logger.py --gpus 0,1 --sum-gpus --interval-ms 50 --output logs/power_trace.csv

Runs until SIGTERM/SIGINT, flushing every row immediately (so a kill mid-run
loses at most one partial sample, never buffered data).

--sum-gpus (for tensor-parallel runs spanning multiple GPUs): collapses all
--gpus into ONE row per sample (power/energy summed across GPUs, temp = max),
instead of one row per GPU per sample. Output CSV schema is then identical to
a single-GPU trace (gpu_index becomes the literal --gpus string, e.g. "0+1"),
so every existing downstream analyze_*/cf_*/plot_* script -- which assumes one
power_w value per timestamp -- works unchanged on a TP>1 trace with no
per-script multi-GPU-awareness needed. Without this flag (default), behavior
is unchanged: one row per GPU per sample, as before.
"""
import argparse
import csv
import signal
import sys
import time

import pynvml

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0", help="comma-separated GPU indices")
    ap.add_argument("--interval-ms", type=int, default=50)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sum-gpus", action="store_true",
                     help="write one summed row per sample instead of one row per GPU "
                          "(for TP>1 runs; see module docstring)")
    args = ap.parse_args()

    gpu_idxs = [int(x) for x in args.gpus.split(",")]

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    pynvml.nvmlInit()
    handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_idxs}

    interval_s = args.interval_ms / 1000.0

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["wall_time", "gpu_index", "power_w", "energy_mj", "temp_c"])
        f.flush()
        while not _stop:
            t = time.time()
            samples = []
            for i, h in handles.items():
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                    temp_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as e:
                    print(f"[power_logger] NVML error on gpu {i}: {e}", file=sys.stderr)
                    continue
                samples.append((i, power_w, energy_mj, temp_c))
            if args.sum_gpus:
                if samples:
                    power_sum = sum(s[1] for s in samples)
                    energy_sum = sum(s[2] for s in samples)
                    temp_max = max(s[3] for s in samples)
                    w.writerow([f"{t:.6f}", args.gpus.replace(",", "+"),
                                f"{power_sum:.2f}", energy_sum, temp_max])
            else:
                for i, power_w, energy_mj, temp_c in samples:
                    w.writerow([f"{t:.6f}", i, f"{power_w:.2f}", energy_mj, temp_c])
            f.flush()
            time.sleep(interval_s)

    pynvml.nvmlShutdown()
    print(f"[power_logger] stopped, wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
