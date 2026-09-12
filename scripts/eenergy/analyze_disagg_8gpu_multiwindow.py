"""Item 2 robustness check: is the peak-power reduction demonstrated at a 15s max
windowed-average robust to the window length a real demand charge actually uses (15 min)? Reuses
analyze_disagg_8gpu_combined_power.py's corrected (average-within-bin, not sum) combine() and
max_windowed_avg(), just sweeping window_s across 15s/60s/300s/900s instead of a single 15s
value. Run against the 1-hour-duration battery (run_disagg_8gpu_1hour_duration.sh), not the
short flagship battery -- a 15-minute window needs a trace substantially longer than 15 minutes
to have somewhere to slide, which the original ~200-380s batteries do not.
"""
import sys

from analyze_disagg_8gpu_combined_power import combine, max_windowed_avg

WINDOWS_S = [15.0, 60.0, 300.0, 900.0]


def main():
    outname = sys.argv[1] if len(sys.argv) > 1 else "disagg_8gpu_1hr_baseline"
    times, powers = combine(f"logs/{outname}_power_prefill.csv", f"logs/{outname}_power_decode.csv")
    duration_s = times[-1] - times[0]
    mean_pow = sum(powers) / len(powers)
    print(f"{outname}: duration_s={duration_s:.1f} combined_mean_power_w={mean_pow:.1f}")
    for w in WINDOWS_S:
        if w > duration_s:
            print(f"  window={w:.0f}s: SKIPPED (trace duration {duration_s:.1f}s < window)")
            continue
        peak = max_windowed_avg(times, powers, w)
        print(f"  window={w:.0f}s: max_windowed_avg_w={peak:.1f}")


if __name__ == "__main__":
    sys.exit(main())
