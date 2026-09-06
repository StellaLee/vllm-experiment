"""Checks whether Share_power's max(ramp_rate, 0) clamp creates a measurable "reignition"
bias: does the router preferentially dispatch new requests onto replicas that are currently
falling fast (where the clamp masks real negative momentum, scoring them identically to a
truly idle/flat replica), and if so, do those replicas subsequently ramp back up more than
ones that were genuinely flat at dispatch time?

For every dispatch event (from the assignment log), reconstructs the replica's own ramp rate
in the window just before dispatch (the same quantity Share_power's clamp would have seen),
buckets it into falling / flat / rising, then measures the replica's forward power rise in a
window just after dispatch. Compares the forward-rise distribution across buckets.

Usage: python3 check_ramp_clamp_reignition.py <prefix> <arm> [trials...]
Example: python3 check_ramp_clamp_reignition.py closedloopheavypergpu drf_power_tiebreak_full 1 2 3
"""
import csv
import sys
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
FORWARD_WINDOW_S = 3.0   # look this far ahead of dispatch for the "reignition" measurement
FLAT_EPS_WPS = 20.0      # |ramp| <= this counts as "flat" (below ~5% of the smallest per-GPU
                          # ceiling, 359.5 W/s -- well within NVML/sampling noise)


def load_power_series(path):
    """Returns {gpu_index: [(wall_time, power_w), ...]} sorted by time."""
    series = defaultdict(list)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            series[int(row["gpu_index"])].append((float(row["wall_time"]), float(row["power_w"])))
    for g in series:
        series[g].sort()
    return series


def load_assignments(path):
    out = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append((float(row["wall_time"]), int(row["gpu_index"])))
    return out


def ramp_before(series_g, t):
    """Ramp rate using the last two samples at or before t. None if unavailable."""
    prior = [(ts, p) for ts, p in series_g if ts <= t]
    if len(prior) < 2:
        return None
    (t0, p0), (t1, p1) = prior[-2], prior[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    return (p1 - p0) / dt


def forward_rise(series_g, t, window):
    """Max power reached in (t, t+window] minus power at/just-before t. None if unavailable."""
    at_t = [(ts, p) for ts, p in series_g if ts <= t]
    if not at_t:
        return None
    p_at_t = at_t[-1][1]
    fwd = [p for ts, p in series_g if t < ts <= t + window]
    if not fwd:
        return None
    return max(fwd) - p_at_t


def analyze(prefix, arm, trials):
    buckets = {"falling": [], "flat": [], "rising": []}
    n_no_data = 0
    for trial in trials:
        pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
        asn_path = f"{LOGDIR}/{prefix}_assignment_{arm}_t{trial}.csv"
        try:
            series = load_power_series(pow_path)
            assignments = load_assignments(asn_path)
        except FileNotFoundError as e:
            print(f"  trial {trial}: MISSING ({e})")
            continue

        for t, gpu in assignments:
            if gpu not in series:
                continue
            rb = ramp_before(series[gpu], t)
            if rb is None:
                n_no_data += 1
                continue
            fr = forward_rise(series[gpu], t, FORWARD_WINDOW_S)
            if fr is None:
                n_no_data += 1
                continue
            if rb < -FLAT_EPS_WPS:
                bucket = "falling"
            elif rb > FLAT_EPS_WPS:
                bucket = "rising"
            else:
                bucket = "flat"
            buckets[bucket].append(fr)

    print(f"\n=== {prefix} / {arm} (trials {trials}) ===")
    print(f"dispatch events with insufficient data: {n_no_data}")
    for name in ("falling", "flat", "rising"):
        vals = buckets[name]
        if not vals:
            print(f"  {name:8s}: n=0")
            continue
        mean_v = sum(vals) / len(vals)
        vals_sorted = sorted(vals)
        median_v = vals_sorted[len(vals_sorted) // 2]
        print(f"  {name:8s}: n={len(vals):4d}  mean_forward_rise={mean_v:8.1f} W  "
              f"median_forward_rise={median_v:8.1f} W")
    return buckets


if __name__ == "__main__":
    prefix = sys.argv[1]
    arm = sys.argv[2]
    trials = [int(x) for x in sys.argv[3:]] if len(sys.argv) > 3 else [1, 2, 3]
    analyze(prefix, arm, trials)
