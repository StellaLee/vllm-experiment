"""Directly inspects the top-K largest per-GPU ramp values feeding the fleet-aggregate
p99_ramp statistic, per arm -- tests whether drf_no_power's tail is worse because of a few
larger-magnitude events, not more frequent ones."""
import csv
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "closedloopheavylongpergpu"

TRIALS = {
    "round_robin": (1, 2, 3, 4, 5, 6),
    "compute_only": (1, 2, 3, 4, 5, 6),
    "load_only": (1, 2, 3, 4, 5, 6),
    "drf_no_power": (1, 2, 3),
}


def fleet_ramps(path):
    """Fleet-aggregate power(t) (summed across GPUs per poll), then ramp between
    consecutive polls -- matches ramp_stats() in compare_closedloopheavy_duration.py."""
    by_time = defaultdict(float)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            by_time[float(row["wall_time"])] += float(row["power_w"])
    series = sorted(by_time.items())
    ramps = []
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        dt = t1 - t0
        if dt > 0:
            ramps.append(abs(p1 - p0) / dt)
    return ramps


for arm, trials in TRIALS.items():
    all_ramps = []
    for trial in trials:
        path = f"{LOGDIR}/{PREFIX}_power_trace_{arm}_t{trial}.csv"
        all_ramps.extend(fleet_ramps(path))
    all_ramps.sort()
    top10 = all_ramps[-10:][::-1]
    p99_idx = int(len(all_ramps) * 0.99)
    print(f"{arm:<16} n={len(all_ramps):6d}  p99={all_ramps[p99_idx]:8.1f}  "
          f"max={all_ramps[-1]:8.1f}  top10={[round(x,1) for x in top10]}")
