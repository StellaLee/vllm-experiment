import csv
import glob
import json
import statistics

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "round_robin"]

def load_records(path):
    """TBT_mean here = mean across requests of each request's own MAX inter-token gap
    (excluding the first tbt_ms entry, a ttft-adjacent artifact) -- matches this project's
    established TBT convention (tracks worst-observed-per-request interference stall, not
    steady-state per-token latency; confirmed by matching the old reported ~95ms scale)."""
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
    """Fleet-aggregate power(t): sum power across all GPUs within each simultaneous poll
    (rows sharing the same wall_time are one poll cycle across all replicas)."""
    from collections import defaultdict
    by_time = defaultdict(float)
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            by_time[float(row["wall_time"])] += float(row["power_w"])
    return sorted(by_time.items())

def ramp_stats(fleet_series):
    peak = max(p for _, p in fleet_series) if fleet_series else 0.0
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

print(f"{'arm':<28} {'peak_power':>16} {'mean_ramp':>14} {'p99_ramp':>16} {'TTFT_mean':>14} {'TBT_mean':>14}")
for arm in ARMS:
    peaks, mean_ramps, p99_ramps, ttft_means, tbt_means = [], [], [], [], []
    for trial in [1, 2, 3]:
        rec_path = f"/root/pli/vllm-experiment/logs/burstgptpergpu_records_{arm}_t{trial}.jsonl"
        pow_path = f"/root/pli/vllm-experiment/logs/burstgptpergpu_power_trace_{arm}_t{trial}.csv"
        ttfts, tbts = load_records(rec_path)
        power_rows = load_power(pow_path)
        peak, mean_ramp, p99_ramp = ramp_stats(power_rows)
        peaks.append(peak)
        mean_ramps.append(mean_ramp)
        p99_ramps.append(p99_ramp)
        if ttfts:
            ttft_means.append(sum(ttfts) / len(ttfts))
        if tbts:
            tbt_means.append(sum(tbts) / len(tbts))
    pk_m, pk_s = mstd(peaks)
    mr_m, mr_s = mstd(mean_ramps)
    p99_m, p99_s = mstd(p99_ramps)
    tt_m, tt_s = mstd(ttft_means)
    tb_m, tb_s = mstd(tbt_means)
    print(f"{arm:<28} {pk_m:>8.1f}±{pk_s:<6.1f} {mr_m:>6.1f}±{mr_s:<6.1f} "
          f"{p99_m:>8.1f}±{p99_s:<6.1f} {tt_m:>6.3f}±{tt_s:<6.3f} {tb_m:>6.1f}±{tb_s:<6.1f}")
