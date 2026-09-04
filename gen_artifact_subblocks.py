import csv
import json
import statistics
from collections import defaultdict

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power"]
METRICS = ["peak", "mean_ramp", "p99_ramp", "ttft", "tbt"]

CONDITIONS = [
    ("Heavy/Matched", "openloopwhalelongoutmatchedpergpu", "openloopwhalelongoutmatched", {}),
    ("Light/Cachehit", "cachehitpergpu", "cachehit", {}),
    ("Heavy/Closed-Loop", "closedloopheavypergpu", "closedloopheavy", {}),
    ("Ramp & Route", "rampandroutepergpu", "rampandroute", {"drf_fixed": "eenergy"}),
    ("WildChat", "wildchatnaturalpergpu", "wildchatnatural", {}),
]
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
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd


def aggregate(prefix, arm, trials=(1, 2, 3)):
    peaks, mean_ramps, p99_ramps, ttft_means, tbt_means = [], [], [], [], []
    for trial in trials:
        rec_path = f"{LOGDIR}/{prefix}_records_{arm}_t{trial}.jsonl"
        pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
        ttfts, tbts = load_records(rec_path)
        power_rows = load_power(pow_path)
        peak, mean_ramp, p99_ramp = ramp_stats(power_rows)
        peaks.append(peak); mean_ramps.append(mean_ramp); p99_ramps.append(p99_ramp)
        ttft_means.append(sum(ttfts) / len(ttfts))
        tbt_means.append(sum(tbts) / len(tbts))
    return {
        "peak": mstd(peaks), "mean_ramp": mstd(mean_ramps), "p99_ramp": mstd(p99_ramps),
        "ttft": mstd(ttft_means), "tbt": mstd(tbt_means),
    }


def dominates(a, b):
    all_le = all(a[m][0] <= b[m][0] for m in METRICS)
    any_lt = any(a[m][0] < b[m][0] for m in METRICS)
    return all_le and any_lt


def cell(vals, is_best, is_worst, prec):
    m, s = vals
    cls = "best" if is_best else ("worst" if is_worst else "")
    cls_attr = f' class="{cls}"' if cls else ""
    return f'<td{cls_attr}>{m:.{prec}f} <span class="std">±{s:.{prec}f}</span></td>'


PREC = {"peak": 1, "mean_ramp": 1, "p99_ramp": 0, "ttft": 3, "tbt": 1}

for label, new_prefix, old_prefix, overrides in CONDITIONS:
    old_vals, new_vals = {}, {}
    for arm in ARMS:
        old_p = overrides.get(arm, old_prefix)
        old_vals[arm] = aggregate(old_p, arm)
        new_vals[arm] = aggregate(new_prefix, arm)

    print(f"\n<!-- ===== {label} pergpu subblock ===== -->")
    rows = []
    for arm in ARMS:
        v = new_vals[arm]
        best = {m: min(new_vals[a][m][0] for a in ARMS) for m in METRICS}
        worst = {m: max(new_vals[a][m][0] for a in ARMS) for m in METRICS}
        arm_cls = "arm-name n-featured" if arm == "drf_power_tiebreak_full" else "arm-name"
        tds = "".join(
            cell(v[m], v[m][0] == best[m], v[m][0] == worst[m], PREC[m]) for m in METRICS
        )
        rows.append(f'            <tr><td class="{arm_cls}">{arm}</td>{tds}</tr>')
    table_html = "\n".join(rows)

    old_doms = [f"{a} dominates {b}" for a in ARMS for b in ARMS if a != b and dominates(old_vals[a], old_vals[b])]
    new_doms = [f"{a} dominates {b}" for a in ARMS for b in ARMS if a != b and dominates(new_vals[a], new_vals[b])]
    print(f"OLD dominance: {old_doms}")
    print(f"NEW dominance: {new_doms}")
    print(table_html)
