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

Runs until SIGTERM/SIGINT, flushing every row immediately (so a kill mid-run
loses at most one partial sample, never buffered data).
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
            for i, h in handles.items():
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    energy_mj = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
                    temp_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as e:
                    print(f"[power_logger] NVML error on gpu {i}: {e}", file=sys.stderr)
                    continue
                w.writerow([f"{t:.6f}", i, f"{power_w:.2f}", energy_mj, temp_c])
            f.flush()
            time.sleep(interval_s)

    pynvml.nvmlShutdown()
    print(f"[power_logger] stopped, wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
