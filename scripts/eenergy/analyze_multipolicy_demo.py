"""Analysis for the 6-arm peak-shaving multipolicy demo battery (cap=1800W, window=15s,
Heavy/CL-long). Computes, per arm: mean fleet power, max rolling-15s-window-average power
(the metric the gate actually targets) vs the 1800W cap, and TTFT distribution."""
import json
import sys
from collections import defaultdict

CAP_W = 1800.0
WINDOW_S = 15.0
PREFIX = "closedloopheavylongpergpu"
OUTNAME = "peak_shaving_multipolicy_demo"
ARMS = [
    "round_robin_no_gate", "round_robin_gated",
    "lmetric_no_gate", "lmetric_gated",
    "drf_no_power_no_gate", "drf_no_power_gated",
]


def load_fleet_power(path):
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
    n_records = 0
    n_failed = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_records += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            ttft = rec.get("ttft")
            if isinstance(ttft, (int, float)):
                ttfts.append(ttft)
            else:
                n_failed += 1
    return n_records, n_failed, sorted(ttfts)


def main():
    print(f"{'arm':<24} {'n_rec':>6} {'n_fail':>7} {'mean_ttft':>10} {'p50_ttft':>9} {'p95_ttft':>9} {'max_ttft':>9} {'mean_pow_w':>11} {'max_win15_w':>12} {'over_cap':>9}")
    for arm in ARMS:
        power_path = f"logs/{PREFIX}_power_trace_{OUTNAME}_{arm}.csv"
        records_path = f"logs/{PREFIX}_records_{OUTNAME}_{arm}.jsonl"
        try:
            times, powers = load_fleet_power(power_path)
        except FileNotFoundError:
            print(f"{arm:<24} MISSING power trace")
            continue
        mean_pow = sum(powers) / len(powers) if powers else float("nan")
        max_win = max_windowed_avg(times, powers, WINDOW_S)
        try:
            n_records, n_failed, ttfts = load_ttft(records_path)
        except FileNotFoundError:
            n_records, n_failed, ttfts = 0, 0, []
        mean_ttft = sum(ttfts) / len(ttfts) if ttfts else float("nan")
        p50 = pctile(ttfts, 0.50)
        p95 = pctile(ttfts, 0.95)
        max_ttft = ttfts[-1] if ttfts else float("nan")
        over = "YES" if max_win > CAP_W else "no"
        print(f"{arm:<24} {n_records:>6} {n_failed:>7} {mean_ttft:>10.3f} {p50:>9.3f} {p95:>9.3f} {max_ttft:>9.3f} {mean_pow:>11.1f} {max_win:>12.1f} {over:>9}")


if __name__ == "__main__":
    sys.exit(main())
