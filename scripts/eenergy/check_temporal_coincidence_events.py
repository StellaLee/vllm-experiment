"""Counts actual temporal coincidence events (polls where 2+ replicas are simultaneously
over their own calibrated ramp ceiling) from the power traces -- tests whether drf_no_power
produces MORE of these than compute_only/round_robin despite spreading whales more evenly by
total count, which would explain its worse p99_ramp via genuine coincident-ramp clustering
rather than raw placement imbalance."""
import csv
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "closedloopheavylongpergpu"
RAMP_CEILING = {"2": 450.2, "3": 509.8, "4": 449.6, "5": 409.2, "6": 512.9, "7": 359.5}

TRIALS = {
    "round_robin": (1, 2, 3, 4, 5, 6),
    "compute_only": (1, 2, 3, 4, 5, 6),
    "load_only": (1, 2, 3, 4, 5, 6),
    "drf_no_power": (1, 2, 3),
}


def load_power_series(path):
    series = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            series[row["gpu_index"]].append((float(row["wall_time"]), float(row["power_w"])))
    for g in series:
        series[g].sort()
    return series


def count_coincidence_polls(path):
    series = load_power_series(path)
    # build ramp rate per GPU at each of its own timestamps, then bucket by rounded wall_time
    # to align across GPUs (polls happen close together, not perfectly simultaneous)
    ramp_by_time = defaultdict(dict)  # rounded_time -> {gpu: ramp}
    for gpu, pts in series.items():
        ceiling = RAMP_CEILING[gpu]
        for (t0, p0), (t1, p1) in zip(pts, pts[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            ramp = (p1 - p0) / dt
            share_power = max(ramp, 0.0) / ceiling
            # round to nearest 0.1s to align near-simultaneous polls across GPUs
            bucket = round(t1, 1)
            ramp_by_time[bucket][gpu] = share_power

    n_polls = len(ramp_by_time)
    n_coincidence = sum(1 for shares in ramp_by_time.values()
                         if sum(1 for s in shares.values() if s > 1.0) >= 2)
    return n_polls, n_coincidence


for arm, trials in TRIALS.items():
    total_polls, total_coincidence = 0, 0
    for trial in trials:
        path = f"{LOGDIR}/{PREFIX}_power_trace_{arm}_t{trial}.csv"
        polls, coincidence = count_coincidence_polls(path)
        total_polls += polls
        total_coincidence += coincidence
    rate = total_coincidence / total_polls if total_polls else 0
    print(f"{arm:<16} polls={total_polls:6d}  coincidence_polls={total_coincidence:5d}  "
          f"rate={rate:.3%}  (n={len(trials)} trials)")
