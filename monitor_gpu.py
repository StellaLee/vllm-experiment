#!/usr/bin/env python3
"""Poll nvidia-smi every 0.5s and write a JSON timeseries. Run with --output path."""
import argparse, json, signal, subprocess, sys, time

INTERVAL = 0.5
QUERY = "power.draw,memory.used,utilization.gpu,temperature.gpu"
FMT = "csv,noheader,nounits"

records = []
running = True

def _stop(sig, frame):
    global running
    running = False

def poll():
    out = subprocess.check_output(
        ["nvidia-smi", f"--query-gpu={QUERY}", f"--format={FMT}"],
        text=True
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "ts": time.time(),
        "power_w": float(parts[0]),
        "mem_mib": int(parts[1]),
        "util_pct": int(parts[2]),
        "temp_c": int(parts[3]),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Path to write JSON timeseries")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    print(f"[monitor_gpu] polling every {INTERVAL}s → {args.output}", flush=True)
    while running:
        try:
            records.append(poll())
        except Exception as e:
            print(f"[monitor_gpu] poll error: {e}", file=sys.stderr)
        time.sleep(INTERVAL)

    with open(args.output, "w") as f:
        json.dump({"interval_s": INTERVAL, "records": records}, f, indent=2)
    print(f"[monitor_gpu] saved {len(records)} records to {args.output}", flush=True)

if __name__ == "__main__":
    main()
