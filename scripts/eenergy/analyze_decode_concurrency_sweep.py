"""Analysis for the decode-concurrency power sweep spike. For each phase (concurrency
2/8/32), slices the single continuous decode-GPU power trace by that phase's recorded
start/end timestamps and reports mean power -- if decode power is genuinely concurrency-
sensitive, mean power should rise clearly across phases; if it's flat, that confirms decode
is closer to a "GPU busy or not" step function than a concurrency-proportional draw."""
import sys

OUTNAME = "decode_concurrency_power_sweep"
PHASES = [("c02", 2), ("c08", 8), ("c32", 32)]


def load_power(path):
    times, powers = [], []
    with open(path) as f:
        next(f)  # header
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            times.append(float(parts[0]))
            powers.append(float(parts[2]))
    return times, powers


def mean_power_in_window(times, powers, t_lo, t_hi):
    vals = [p for t, p in zip(times, powers) if t_lo <= t <= t_hi]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    with open(f"logs/{OUTNAME}_timing.txt") as f:
        timing = {}
        for line in f:
            k, v = line.strip().split("=")
            timing[k] = float(v)

    times, powers = load_power(f"logs/{OUTNAME}_power_decode.csv")
    idle_before = mean_power_in_window(times, powers, times[0], timing["phase_c02_start_epoch"])
    print(f"idle_before_power_w={idle_before:.1f}")

    prev_mean = None
    for phase, concurrency in PHASES:
        t_lo = timing[f"phase_{phase}_start_epoch"]
        t_hi = timing[f"phase_{phase}_end_epoch"]
        mean_pow = mean_power_in_window(times, powers, t_lo, t_hi)
        delta = "" if prev_mean is None else f"  (delta vs prev: {mean_pow - prev_mean:+.1f}W)"
        print(f"phase={phase} concurrency={concurrency:3d} mean_power_w={mean_pow:.1f}{delta}")
        prev_mean = mean_pow


if __name__ == "__main__":
    sys.exit(main())
