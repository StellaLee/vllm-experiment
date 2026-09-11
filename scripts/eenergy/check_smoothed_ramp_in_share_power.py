"""Reconstructs the exact raw vs smoothed ramp_rate_w_per_s series the router computed live
during the drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp reference battery (Heavy/
Closed-Loop long, 3 trials), by replaying ramp.update_ramp_state() over each GPU's power_trace
CSV in chronological order (power_poll_loop calls this every 0.5s per replica, exactly what's
in the trace). Reports how much the EMA (alpha=0.3) actually damps noise in the ramp signal
that feeds Share_power = ramp_rate / ramp_ceiling, and how much that changes the fraction of
samples classified "elevated" (share_power_level > 1.0, i.e. at/above ceiling) -- the input to
coincidence_ceiling_factor -- since flip-flopping in/out of "elevated" on noise is exactly the
kind of instability smoothing is meant to fix.
"""
import csv
import glob
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, "scripts/eenergy/router")
from ramp import update_ramp_state  # noqa: E402


class _State:
    def __init__(self):
        self.last_power_w = 0.0
        self.last_power_ts = 0.0
        self.ramp_rate_w_per_s = 0.0
        self.smoothed_ramp_rate_w_per_s = 0.0


RAMP_CEILING = {2: 450.2, 3: 509.8, 4: 449.6, 5: 409.2, 6: 512.9, 7: 359.5}

PREFIX = "closedloopheavylongpergpu"
OUTNAME = "drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp"


def load_trace(path):
    by_gpu = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_gpu[int(row["gpu_index"])].append(
                (float(row["wall_time"]), float(row["power_w"])))
    for g in by_gpu:
        by_gpu[g].sort()
    return by_gpu


def main():
    files = sorted(glob.glob(f"logs/{PREFIX}_power_trace_{OUTNAME}_t*.csv"))
    if not files:
        print("no power trace files found")
        return

    all_raw_rel_std = []
    all_smoothed_rel_std = []
    all_raw_elevated_frac = []
    all_smoothed_elevated_frac = []
    all_raw_flip_rate = []
    all_smoothed_flip_rate = []

    for path in files:
        trial = path.split("_t")[-1].replace(".csv", "")
        by_gpu = load_trace(path)
        print(f"\n=== trial {trial} ({path}) ===")
        for gpu, samples in sorted(by_gpu.items()):
            ceiling = RAMP_CEILING[gpu]
            state = _State()
            raw_series = []
            smoothed_series = []
            for ts, power_w in samples:
                update_ramp_state(state, power_w, ts, smoothing_alpha=0.3)
                raw_series.append(state.ramp_rate_w_per_s)
                smoothed_series.append(state.smoothed_ramp_rate_w_per_s)

            # skip the very first sample (both fields still 0.0, not a real reading)
            raw_series = raw_series[1:]
            smoothed_series = smoothed_series[1:]
            if len(raw_series) < 10:
                continue

            raw_std = statistics.pstdev(raw_series)
            smoothed_std = statistics.pstdev(smoothed_series)
            raw_mean_abs = statistics.mean(abs(x) for x in raw_series) or 1e-9
            smoothed_mean_abs = statistics.mean(abs(x) for x in smoothed_series) or 1e-9

            raw_share = [x / ceiling for x in raw_series]
            smoothed_share = [x / ceiling for x in smoothed_series]
            raw_elevated = [s > 1.0 for s in raw_share]
            smoothed_elevated = [s > 1.0 for s in smoothed_share]
            raw_elevated_frac = sum(raw_elevated) / len(raw_elevated)
            smoothed_elevated_frac = sum(smoothed_elevated) / len(smoothed_elevated)

            def flip_rate(bools):
                if len(bools) < 2:
                    return 0.0
                flips = sum(1 for a, b in zip(bools, bools[1:]) if a != b)
                return flips / (len(bools) - 1)

            raw_flips = flip_rate(raw_elevated)
            smoothed_flips = flip_rate(smoothed_elevated)

            print(f"  gpu{gpu} (ceiling={ceiling:.1f}): "
                  f"raw ramp std={raw_std:7.2f} W/s (rel {raw_std/raw_mean_abs*100:5.1f}%)  "
                  f"smoothed std={smoothed_std:7.2f} W/s (rel {smoothed_std/smoothed_mean_abs*100:5.1f}%)  "
                  f"|| elevated-frac raw={raw_elevated_frac*100:5.2f}% smoothed={smoothed_elevated_frac*100:5.2f}%  "
                  f"|| flip-rate raw={raw_flips*100:5.2f}% smoothed={smoothed_flips*100:5.2f}%")

            all_raw_rel_std.append(raw_std / raw_mean_abs * 100)
            all_smoothed_rel_std.append(smoothed_std / smoothed_mean_abs * 100)
            all_raw_elevated_frac.append(raw_elevated_frac * 100)
            all_smoothed_elevated_frac.append(smoothed_elevated_frac * 100)
            all_raw_flip_rate.append(raw_flips * 100)
            all_smoothed_flip_rate.append(smoothed_flips * 100)

    print("\n=== SUMMARY (mean across all gpu x trial series) ===")
    print(f"raw ramp relative std:      {statistics.mean(all_raw_rel_std):6.1f}%")
    print(f"smoothed ramp relative std: {statistics.mean(all_smoothed_rel_std):6.1f}%")
    print(f"raw elevated-frac (share_power>1.0):      {statistics.mean(all_raw_elevated_frac):6.2f}%")
    print(f"smoothed elevated-frac (share_power>1.0): {statistics.mean(all_smoothed_elevated_frac):6.2f}%")
    print(f"raw flip-rate (elevated<->normal per sample):      {statistics.mean(all_raw_flip_rate):6.2f}%")
    print(f"smoothed flip-rate (elevated<->normal per sample): {statistics.mean(all_smoothed_flip_rate):6.2f}%")


if __name__ == "__main__":
    main()
