import csv
import json
import statistics
from collections import defaultdict
from itertools import permutations

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power", "round_robin"]
METRICS = ["peak", "mean_ramp", "p99_ramp", "ttft", "tbt"]  # all lower-is-better

CONDITIONS = {
    "Heavy/Matched":     ("openloopwhalelongoutmatchedpergpu", "openloopwhalelongoutmatched", {}),
    "Light/Cachehit":    ("cachehitpergpu", "cachehit", {}),
    "Heavy/Closed-Loop": ("closedloopheavypergpu", "closedloopheavy", {}),
    "Ramp & Route":      ("rampandroutepergpu", "rampandroute", {"drf_fixed": "eenergy"}),
    "WildChat":          ("wildchatnaturalpergpu", "wildchatnatural", {}),
}
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
    return {"peak": m(peaks), "mean_ramp": m(mean_ramps), "p99_ramp": m(p99_ramps),
            "ttft": m(ttft_means), "tbt": m(tbt_means)}


def dominates(a, b):
    """True if a weakly dominates b on all metrics with >=1 strict."""
    all_le = all(a[m] <= b[m] for m in METRICS)
    any_lt = any(a[m] < b[m] for m in METRICS)
    return all_le and any_lt


for cond, (new_prefix, old_prefix, overrides) in CONDITIONS.items():
    old_vals, new_vals = {}, {}
    for arm in ARMS:
        old_p = overrides.get(arm, old_prefix)
        try:
            old_vals[arm] = aggregate(old_p, arm)
        except FileNotFoundError:
            pass
        new_vals[arm] = aggregate(new_prefix, arm)

    print(f"\n=== {cond} ===")
    for label, vals in [("OLD", old_vals), ("NEW", new_vals)]:
        present_arms = [a for a in ARMS if a in vals]
        skipped = [a for a in ARMS if a not in vals]
        doms = []
        for a, b in permutations(present_arms, 2):
            if dominates(vals[a], vals[b]):
                doms.append(f"{a} DOMINATES {b}")
        note = f" (skipped, no data: {', '.join(skipped)})" if skipped else ""
        print(f"  {label}: {'; '.join(doms) if doms else 'no dominance (incomparable)'}{note}")
