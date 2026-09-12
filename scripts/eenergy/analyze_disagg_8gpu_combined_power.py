"""Combines the prefill-pool and decode-pool power CSVs into one facility-wide (all 8 GPU)
series -- a real demand charge bills on TOTAL facility peak, not per-pool, so this is the
number the electricity-cost analysis needs, not Tables 2/6's per-pool figures. Prefill and
decode power_logger.py sidecars are separate processes with independently-timestamped
samples, so exact-timestamp matching (as used within a single pool's own CSV) doesn't apply
across pools -- this bins both pools' samples into 1s buckets and sums within each bucket."""
import sys
from collections import defaultdict

WINDOW_S = 15.0


def load_binned(path, bin_s=1.0):
    """Returns {bin_time: mean_fleet_power_in_that_bin}. Each bin typically contains samples
    from all 4 GPUs in this pool across several ~50ms ticks -- must AVERAGE across those
    ticks (not sum), after first summing across the 4 simultaneous GPUs at each tick, else
    the bin's power gets inflated by however many ticks fall in it (~20 for a 1s bin at
    50ms sampling), not just correctly combining the 4 GPUs."""
    tick_totals = defaultdict(float)  # exact timestamp -> summed power across that pool's 4 GPUs
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            t = float(parts[0])
            p = float(parts[2])
            tick_totals[t] += p

    bin_sums = defaultdict(float)
    bin_counts = defaultdict(int)
    for t, fleet_p in tick_totals.items():
        b = round(t / bin_s) * bin_s
        bin_sums[b] += fleet_p
        bin_counts[b] += 1
    return {b: bin_sums[b] / bin_counts[b] for b in bin_sums}


def combine(prefill_path, decode_path, bin_s=1.0):
    prefill = load_binned(prefill_path, bin_s)
    decode = load_binned(decode_path, bin_s)
    all_bins = sorted(set(prefill) | set(decode))
    times = all_bins
    powers = [prefill.get(b, 0.0) + decode.get(b, 0.0) for b in all_bins]
    return times, powers


def max_windowed_avg(times, powers, window_s):
    n = len(times)
    j = 0
    window_sum = 0.0
    best = 0.0
    for i in range(n):
        window_sum += powers[i]
        while times[i] - times[j] > window_s:
            window_sum -= powers[j]
            j += 1
        span = times[i] - times[j]
        if span <= 0:
            continue
        avg = window_sum / (i - j + 1)
        if avg > best:
            best = avg
    return best


def main():
    outname = sys.argv[1] if len(sys.argv) > 1 else "disagg_8gpu_baseline"
    times, powers = combine(f"logs/{outname}_power_prefill.csv", f"logs/{outname}_power_decode.csv")
    mean_pow = sum(powers) / len(powers)
    max_win = max_windowed_avg(times, powers, WINDOW_S)
    print(f"{outname}: combined_mean_power_w={mean_pow:.1f} "
          f"combined_max_windowed_avg_{int(WINDOW_S)}s_w={max_win:.1f}")


if __name__ == "__main__":
    sys.exit(main())
