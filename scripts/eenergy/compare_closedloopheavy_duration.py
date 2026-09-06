"""Compares Heavy/Closed-Loop's original ~51-53s active window (closedloopheavypergpu_*)
against the 5-minute duration-extension battery (closedloopheavylongpergpu_*), same 5 arms,
same per-GPU-calibrated ceiling. Reports both aggregates side by side and checks whether the
headline dominance relationships (proposed rule / weighted_sum vs. lmetric_power; round_robin
vs. the 4 scored arms) hold at the longer duration."""
import csv
import json
from itertools import permutations

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power", "round_robin"]
METRICS = ["peak", "mean_ramp", "p99_ramp", "ttft", "tbt"]
LOGDIR = "/root/pli/vllm-experiment/logs"


def load_records(path):
    ttfts, tbt_maxes = [], []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("ttft") is not None:
                ttfts.append(r["ttft"])
            vals = r.get("tbt_ms", [])[1:]
            if vals:
                tbt_maxes.append(max(vals))
    return ttfts, tbt_maxes


def load_power(path):
    from collections import defaultdict
    by_time = defaultdict(float)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            by_time[float(row["wall_time"])] += float(row["power_w"])
    return sorted(by_time.items())


def ramp_stats(fleet_series):
    if not fleet_series:
        return 0.0, 0.0, 0.0
    peak = max(p for _, p in fleet_series)
    ramps = []
    for (t0, p0), (t1, p1) in zip(fleet_series, fleet_series[1:]):
        dt = t1 - t0
        if dt > 0:
            ramps.append(abs(p1 - p0) / dt)
    ramps.sort()
    mean_ramp = sum(ramps) / len(ramps) if ramps else 0.0
    p99_ramp = ramps[int(len(ramps) * 0.99)] if ramps else 0.0
    return peak, mean_ramp, p99_ramp


def aggregate(prefix, arm, trials=(1, 2, 3)):
    peaks, mean_ramps, p99_ramps, ttft_means, tbt_means = [], [], [], [], []
    for trial in trials:
        rec_path = f"{LOGDIR}/{prefix}_records_{arm}_t{trial}.jsonl"
        pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
        ttfts, tbts = load_records(rec_path)
        power_rows = load_power(pow_path)
        peak, mean_ramp, p99_ramp = ramp_stats(power_rows)
        peaks.append(peak)
        mean_ramps.append(mean_ramp)
        p99_ramps.append(p99_ramp)
        ttft_means.append(sum(ttfts) / len(ttfts))
        tbt_means.append(sum(tbts) / len(tbts))
    def m(vals):
        return sum(vals) / len(vals)
    def sd(vals):
        if len(vals) < 2:
            return 0.0
        mu = m(vals)
        return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return {
        "peak": (m(peaks), sd(peaks)), "mean_ramp": (m(mean_ramps), sd(mean_ramps)),
        "p99_ramp": (m(p99_ramps), sd(p99_ramps)), "ttft": (m(ttft_means), sd(ttft_means)),
        "tbt": (m(tbt_means), sd(tbt_means)),
    }


def dominates(a, b):
    all_le = all(a[m][0] <= b[m][0] for m in METRICS)
    any_lt = any(a[m][0] < b[m][0] for m in METRICS)
    return all_le and any_lt


for label, prefix in [("SHORT (~51-53s)", "closedloopheavypergpu"), ("LONG (~225-232s)", "closedloopheavylongpergpu")]:
    print(f"\n{'='*90}\n{label}: {prefix}\n{'='*90}")
    vals = {}
    for arm in ARMS:
        vals[arm] = aggregate(prefix, arm)
    print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
    for arm in ARMS:
        v = vals[arm]
        print(f"{arm:<26}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
              f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    print("\nDominance (scored arms only, drf_fixed/drf_power_tiebreak_full/weighted_sum vs lmetric_power):")
    scored = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power"]
    for a, b in permutations(scored, 2):
        if dominates(vals[a], vals[b]):
            print(f"  {a} DOMINATES {b}")

    print("\nround_robin vs each scored arm:")
    for arm in scored:
        if dominates(vals["round_robin"], vals[arm]):
            print(f"  round_robin DOMINATES {arm}")
        elif dominates(vals[arm], vals["round_robin"]):
            print(f"  {arm} DOMINATES round_robin")
        else:
            print(f"  round_robin vs {arm}: incomparable")
