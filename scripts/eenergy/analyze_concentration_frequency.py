#!/usr/bin/env python3
"""Concentration-frequency + aggregate-ramp analysis behind paper-eenergy's §5 "Does
per-replica discipline reduce fleet-wide concentration?" paragraph and Table 6.

Motivation: Theorem 4 (threshold-safety) bounds the replica that receives a request, at the
moment of that decision -- it says nothing formally about the fleet-aggregate ramp reported
in Table 1, which is a sum across replicas and depends on cross-replica correlation
("concentration", the same concept already used in router/adaptive_ceiling.py's round-filtered
calibration), not on any one replica's level alone. There is no theorem closing this gap
without additional, unverified workload-correlation assumptions (see the paper's §6
discussion). This script instead checks it empirically, on a longer companion run than
Table 1's headline trials (6 trials, ~226s each, same Heavy/Closed-Loop condition and
per-GPU calibration -- prefix `closedloopheavylongpergpu` in the remote experiment logs,
vs. Table 1's `closedloopheavypergpu`, 3 trials, ~31s each).

"Concentration" uses the EXACT same definition as router/adaptive_ceiling.py's
observe_round(): ramp clamped to non-negative (matching Share_power's own clamp -- a falling
ramp is never a routing-relevant pressure signal), "elevated" = clamped ramp > floor,
"concentration" = >=2 replicas elevated simultaneously in the same decision round.

Data source: raw per-GPU power traces (wall_time, gpu_index, power_w, energy_mj, temp_c),
50ms-sampled NVML sidecar CSVs, live only on the remote experiment server
(183.147.142.123:/root/pli/vllm-experiment/logs/closedloopheavylongpergpu_power_trace_
<arm>_t<trial>.csv) -- not checked into this repo (large, and this script is the reproducible
artifact, not the data). Pull the 30 files (5 arms x 6 trials) via scp into a local directory
and pass that directory as sys.argv[1] (default: current directory).

Sanity-checked against Table 1 before trusting this dataset for anything new: the SHORT
(closedloopheavypergpu, 3-trial) version of this same script's ramp_stats_abs() reproduces
Table 1's published peak/mean_ramp/p99_ramp numbers exactly (mean values bit-for-bit; std
differs only by the population-vs-sample-stdev convention, consistent with n=3 sample std).
"""
import csv
import glob
import os
import sys
import statistics

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power", "round_robin"]
TRIALS = (1, 2, 3, 4, 5, 6)
PREFIX = "closedloopheavylongpergpu"
DEFAULT_FLOOR = 450.0  # matches router/adaptive_ceiling.py's DEFAULT_FLOOR_W_PER_S
THRESHOLDS_TO_CHECK = (150.0, 200.0, 300.0, DEFAULT_FLOOR)


def load_trial(data_dir, arm, trial):
    path = os.path.join(data_dir, f"{PREFIX}_power_trace_{arm}_t{trial}.csv")
    by_time = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = float(row["wall_time"])
            gpu = row["gpu_index"]
            p = float(row["power_w"])
            by_time.setdefault(t, {})[gpu] = p
    times = sorted(by_time)
    # only GPUs present at every timestep (robust to any dropped sidecar reads)
    gpus = set.intersection(*(set(by_time[t]) for t in times))
    series = {g: [(t, by_time[t][g]) for t in times] for g in gpus}
    fleet = [(t, sum(by_time[t].values())) for t in times]
    return series, fleet


def ramp_stats_abs(fleet_series):
    """Matches aggregate_full_comparison.py's ramp_stats(): abs(dP)/dt on the fleet-aggregate
    (bidirectional -- a grid-facing ramp hazard is generally bidirectional, unlike
    Share_power's routing-time clamp; see paper §5's setup paragraph)."""
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


def concentrated_count(per_gpu_series, floor, min_elevated=2):
    """Per-replica ramp clamped to non-negative (matching Share_power / adaptive_ceiling.py),
    NOT abs() -- concentration is specifically about routing-relevant upward pressure."""
    gpus = list(per_gpu_series.keys())
    n_steps = len(per_gpu_series[gpus[0]]) - 1
    n_conc = 0
    for i in range(n_steps):
        elevated = 0
        for g in gpus:
            t0, p0 = per_gpu_series[g][i]
            t1, p1 = per_gpu_series[g][i + 1]
            dt = t1 - t0
            if dt <= 0:
                continue
            ramp = max((p1 - p0) / dt, 0.0)
            if ramp > floor:
                elevated += 1
        if elevated >= min_elevated:
            n_conc += 1
    return n_conc, n_steps


def two_proportion_z(c1, n1, c2, n2):
    """Normal-approximation gut-check, NOT a rigorous test: consecutive decision rounds
    within a trial are not independent draws (a real elevated episode likely spans several
    consecutive rounds), which inflates apparent significance somewhat. The robustness of
    the result across multiple independent threshold choices is the more convincing part of
    this evidence than any single z-value -- see the paper's §5/§6 discussion."""
    p1, p2 = c1 / n1, c2 / n2
    se = (p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) ** 0.5
    return (p2 - p1) / se if se > 0 else float("nan")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    cache = {}
    print("=" * 78)
    print(f"Table 6 numbers (floor={DEFAULT_FLOOR} W/s, mean +/- sample std over 6 trials)")
    print("=" * 78)
    for arm in ARMS:
        peaks, p99s, freqs = [], [], []
        for t in TRIALS:
            series, fleet = load_trial(data_dir, arm, t)
            cache[(arm, t)] = series
            peak, mean_r, p99 = ramp_stats_abs(fleet)
            nc, ns = concentrated_count(series, DEFAULT_FLOOR)
            peaks.append(peak); p99s.append(p99); freqs.append(100 * nc / ns)
        print(f"  {arm:28s} peak={statistics.mean(peaks):7.1f}+-{statistics.stdev(peaks):5.1f}  "
              f"p99_ramp={statistics.mean(p99s):7.1f}+-{statistics.stdev(p99s):5.1f}  "
              f"conc_freq={statistics.mean(freqs):5.2f}+-{statistics.stdev(freqs):4.2f}%")

    print()
    print("=" * 78)
    print("Robustness across independently-chosen elevation thresholds (pooled across trials)")
    print("=" * 78)
    for floor in THRESHOLDS_TO_CHECK:
        totals = {}
        for arm in ARMS:
            tc, tr = 0, 0
            for t in TRIALS:
                nc, ns = concentrated_count(cache[(arm, t)], floor)
                tc += nc; tr += ns
            totals[arm] = (tc, tr)
        lowest_arm = min(totals, key=lambda a: totals[a][0] / totals[a][1])
        print(f"  floor={floor:6.1f} W/s: lowest={lowest_arm} "
              f"({100*totals[lowest_arm][0]/totals[lowest_arm][1]:.2f}%)")
        c1, n1 = totals["drf_power_tiebreak_full"]
        for arm in ARMS:
            if arm == "drf_power_tiebreak_full":
                continue
            c2, n2 = totals[arm]
            z = two_proportion_z(c1, n1, c2, n2)
            print(f"      proposed vs {arm:28s} z={z:.2f}")


if __name__ == "__main__":
    main()
