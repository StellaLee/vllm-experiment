import csv
import glob
import json
import statistics
from collections import defaultdict

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power", "round_robin"]

# condition -> (new_prefix, old_prefix, {arm_override: old_prefix})
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


def mstd(vals):
    if not vals:
        return float("nan"), float("nan")
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def aggregate(prefix, arm, trials=(1, 2, 3)):
    peaks, mean_ramps, p99_ramps, ttft_means, tbt_means = [], [], [], [], []
    missing = []
    for trial in trials:
        rec_path = f"{LOGDIR}/{prefix}_records_{arm}_t{trial}.jsonl"
        pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
        try:
            ttfts, tbts = load_records(rec_path)
            power_rows = load_power(pow_path)
        except FileNotFoundError as e:
            missing.append(str(e))
            continue
        peak, mean_ramp, p99_ramp = ramp_stats(power_rows)
        peaks.append(peak)
        mean_ramps.append(mean_ramp)
        p99_ramps.append(p99_ramp)
        if ttfts:
            ttft_means.append(sum(ttfts) / len(ttfts))
        if tbts:
            tbt_means.append(sum(tbts) / len(tbts))
    return {
        "peak": mstd(peaks),
        "mean_ramp": mstd(mean_ramps),
        "p99_ramp": mstd(p99_ramps),
        "ttft": mstd(ttft_means),
        "tbt": mstd(tbt_means),
        "n": len(peaks),
        "missing": missing,
    }


def fmt(m, s, prec=1):
    if m != m:  # nan
        return "MISSING"
    return f"{m:.{prec}f}±{s:.{prec}f}"


results = {}
for cond, (new_prefix, old_prefix, overrides) in CONDITIONS.items():
    for arm in ARMS:
        old_p = overrides.get(arm, old_prefix)
        old = aggregate(old_p, arm)
        new = aggregate(new_prefix, arm)
        results[(cond, arm)] = (old, new)

for cond in CONDITIONS:
    print(f"\n=== {cond} ===")
    print(f"{'arm':<26}{'':>4}{'peak_power(W)':>20}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>16}{'TBT(ms)':>16}")
    for arm in ARMS:
        old, new = results[(cond, arm)]
        print(f"{arm:<26}{'OLD':>4}"
              f"{fmt(*old['peak']):>20}{fmt(*old['mean_ramp']):>18}{fmt(*old['p99_ramp'],0):>18}"
              f"{fmt(*old['ttft'],3):>16}{fmt(*old['tbt'],1):>16}  (n={old['n']})")
        print(f"{'':<26}{'NEW':>4}"
              f"{fmt(*new['peak']):>20}{fmt(*new['mean_ramp']):>18}{fmt(*new['p99_ramp'],0):>18}"
              f"{fmt(*new['ttft'],3):>16}{fmt(*new['tbt'],1):>16}  (n={new['n']})")
