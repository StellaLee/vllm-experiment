"""Analysis for the 4:4 disaggregated baseline/gated batteries (Tasks 4-5). Reports, per
pool: mean power (the number Task 5 uses to compute a cap) and max windowed-average power
(the metric the gate targets) -- plus TTFT distribution from the shared records file. Reuses
the exact windowed-average/percentile logic from analyze_multipolicy_demo.py, applied to
two pool-level power traces instead of one fleet-level trace."""
import json
import sys
from collections import defaultdict

WINDOW_S = 15.0


def load_pool_power(path):
    by_time = defaultdict(float)
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            t = float(parts[0])
            p = float(parts[2])
            by_time[t] += p
    times = sorted(by_time)
    powers = [by_time[t] for t in times]
    return times, powers


def max_windowed_avg(times, powers, window_s):
    n = len(times)
    j = 0
    window_sum = 0.0
    best = 0.0
    for i in range(n):
        window_sum += powers[i]
        while times[i] - times[j] > window_s:
            window_sum -= powers[j]
            j += 1
        span = times[i] - times[j]
        if span <= 0:
            continue
        avg = window_sum / (i - j + 1)
        if avg > best:
            best = avg
    return best


def pctile(sorted_vals, p):
    if not sorted_vals:
        return float("nan")
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def load_ttft(path):
    ttfts = []
    tpots = []
    n_records = 0
    n_failed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            rec = json.loads(line)
            ttft = rec.get("ttft")
            tpot = rec.get("tpot")
            if isinstance(ttft, (int, float)):
                ttfts.append(ttft)
            else:
                n_failed += 1
            if isinstance(tpot, (int, float)):
                tpots.append(tpot)
    return n_records, n_failed, sorted(ttfts), tpots


def main():
    outname = sys.argv[1] if len(sys.argv) > 1 else "disagg_8gpu_baseline"
    for pool in ("prefill", "decode"):
        times, powers = load_pool_power(f"logs/{outname}_power_{pool}.csv")
        mean_pow = sum(powers) / len(powers)
        max_win = max_windowed_avg(times, powers, WINDOW_S)
        print(f"{pool}: mean_power_w={mean_pow:.1f} max_windowed_avg_{int(WINDOW_S)}s_w={max_win:.1f}")

    n_records, n_failed, ttfts, tpots = load_ttft(f"logs/{outname}_records.jsonl")
    mean_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
    mean_tpot = sum(tpots) / len(tpots) if tpots else float("nan")
    print(f"n_records={n_records} n_failed={n_failed} mean_ttft={mean_ttft:.3f} "
          f"p50_ttft={pctile(ttfts, 0.50):.3f} p95_ttft={pctile(ttfts, 0.95):.3f} "
          f"max_ttft={ttfts[-1] if ttfts else float('nan'):.3f} mean_tpot_ms={mean_tpot*1000:.2f}")


if __name__ == "__main__":
    sys.exit(main())
