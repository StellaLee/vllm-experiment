#!/usr/bin/env python3
"""Facility-meter coincidence metric for the routing-condition power traces: how often are
2+ GPUs ramping in the SAME direction simultaneously (reinforcing at the aggregate meter),
as opposed to classify_power_windows.py's abs()-based per-replica pressure labels (correct
for that module's own purpose -- conditioning per-replica request latency on the router's
own reactive signal, which is itself asymmetric/abs-adjacent -- but wrong for THIS purpose).

Mixed-direction simultaneous ramping (one GPU up, another down) cancels in the signed sum a
utility meter actually measures, so it is deliberately excluded here -- unlike
classify_power_windows.py, which is direction-agnostic on purpose for its own use case.
See scripts/eenergy/README.md and findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md
for the reasoning history.

Usage: POWER_TRACE=logs/eenergy_power_trace_drf.csv GPUS=2,3,4,5,6,7 \
         python3 scripts/eenergy/analyze_coincidence.py
"""
import csv
import os

POWER_TRACE = os.environ.get("POWER_TRACE", "logs/eenergy_power_trace_drf.csv")
GPUS = [int(x) for x in os.environ.get("GPUS", "2,3,4,5,6,7").split(",")]
RAMP_CEILING_W_PER_S = float(os.environ.get("RAMP_CEILING_W_PER_S", 450.0))


def load_power(path, gpu_indices):
    per_gpu = {i: [] for i in gpu_indices}
    with open(path) as f:
        for row in csv.DictReader(f):
            gi = int(row["gpu_index"])
            if gi in per_gpu:
                per_gpu[gi].append((float(row["wall_time"]), float(row["power_w"])))
    for i in per_gpu:
        per_gpu[i].sort()
    return per_gpu


def signed_ramp_by_tick(series, tick_s=0.05):
    """tick index -> signed ramp rate (W/s) for the interval starting at that tick."""
    out = {}
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        out[round(t0 / tick_s)] = (p1 - p0) / dt
    return out


def same_direction_coincidence(per_gpu: dict, ramp_ceiling_w_per_s: float, tick_s: float = 0.05):
    """Returns (total_duration_s, any_pressure_time_s, same_direction_coincidence_time_s,
    max_same_direction_count). "Pressure" per GPU = |signed ramp| > ceiling; "coincidence"
    counts only ticks where 2+ GPUs are pressured in the SAME direction (both up or both
    down), since that's what reinforces rather than cancels at the facility meter."""
    signed = {gi: signed_ramp_by_tick(series, tick_s) for gi, series in per_gpu.items()}
    all_ticks = sorted(set(k for s in signed.values() for k in s))
    if len(all_ticks) < 2:
        return 0.0, 0.0, 0.0, 0

    total_duration = (all_ticks[-1] - all_ticks[0]) * tick_s
    any_pressure_time = 0.0
    coincidence_time = 0.0
    max_same_dir = 0
    prev_tick = None
    for k in all_ticks:
        n_up = sum(1 for s in signed.values() if s.get(k, 0.0) > ramp_ceiling_w_per_s)
        n_down = sum(1 for s in signed.values() if s.get(k, 0.0) < -ramp_ceiling_w_per_s)
        same_dir = max(n_up, n_down)
        max_same_dir = max(max_same_dir, same_dir)
        if prev_tick is not None:
            dt = (k - prev_tick) * tick_s
            if n_up + n_down >= 1:
                any_pressure_time += dt
            if same_dir >= 2:
                coincidence_time += dt
        prev_tick = k
    return total_duration, any_pressure_time, coincidence_time, max_same_dir


def main():
    per_gpu = load_power(POWER_TRACE, GPUS)
    duration, any_pressure, coincidence, max_same_dir = same_direction_coincidence(
        per_gpu, RAMP_CEILING_W_PER_S)
    frac_of_pressure = coincidence / any_pressure if any_pressure else float("nan")
    print(f"{POWER_TRACE}: duration={duration:.1f}s, ramp_ceiling={RAMP_CEILING_W_PER_S} W/s")
    print(f"  max GPUs simultaneously ramping same direction: {max_same_dir} / {len(GPUS)}")
    print(f"  any-GPU pressure time: {any_pressure:.2f}s ({100*any_pressure/duration:.1f}% of run)")
    print(f"  2+ GPU same-direction coincidence time: {coincidence:.2f}s "
          f"({100*frac_of_pressure:.1f}% of all pressure-time)")


if __name__ == "__main__":
    main()
