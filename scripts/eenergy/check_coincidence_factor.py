import csv
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
RAMP_CEILING_PER_GPU = {2: 450.2, 3: 509.8, 4: 449.6, 5: 409.2, 6: 512.9, 7: 359.5}


def coincidence_stats(prefix, arm, trial):
    """Returns (any_pressured_frac, coincidence_pct): any_pressured_frac = fraction of
    poll intervals where >=1 GPU exceeds its own calibrated ramp ceiling; coincidence_pct
    = of those pressured intervals, what fraction have >=2 GPUs pressured simultaneously
    (the fleet-aggregate signal coincidence_ceiling's mechanism targets)."""
    path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    by_gpu = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            gpu = int(row["gpu_index"])
            by_gpu[gpu].append((float(row["wall_time"]), float(row["power_w"])))

    # pressured[t0] = set of gpus pressured in the interval starting at t0
    pressured_by_interval = defaultdict(set)
    all_starts = set()
    for gpu, samples in by_gpu.items():
        samples.sort()
        ceiling = RAMP_CEILING_PER_GPU[gpu]
        for (t0, p0), (t1, p1) in zip(samples, samples[1:]):
            dt = t1 - t0
            if dt <= 0:
                continue
            ramp = abs(p1 - p0) / dt
            all_starts.add(t0)
            if ramp > ceiling:
                pressured_by_interval[t0].add(gpu)

    total_intervals = len(all_starts)
    if total_intervals == 0:
        return 0.0, 0.0
    any_pressured = sum(1 for t0 in all_starts if len(pressured_by_interval.get(t0, ())) >= 1)
    coincident = sum(1 for t0 in all_starts if len(pressured_by_interval.get(t0, ())) >= 2)
    any_frac = any_pressured / total_intervals
    coincidence_pct = (coincident / any_pressured * 100) if any_pressured else 0.0
    return any_frac, coincidence_pct


if __name__ == "__main__":
    import sys
    PREFIX = sys.argv[1] if len(sys.argv) > 1 else "closedloopheavylongpergpu"
    ARM = sys.argv[2] if len(sys.argv) > 2 else "lmetric_power_coincidence_ceiling"
    TRIALS = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else [1, 2, 3, 4, 5, 6]

    vals_any, vals_coinc = [], []
    print(f"{'trial':<8}{'any_pressured_frac':>20}{'coincidence_pct':>18}")
    for t in TRIALS:
        try:
            a, c = coincidence_stats(PREFIX, ARM, t)
        except FileNotFoundError:
            continue
        vals_any.append(a)
        vals_coinc.append(c)
        print(f"{t:<8}{a:>20.4f}{c:>18.2f}")

    def stats(vals):
        mu = sum(vals) / len(vals)
        sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
        return mu, sd

    mu_a, sd_a = stats(vals_any)
    mu_c, sd_c = stats(vals_coinc)
    print(f"\nany_pressured_frac: mean={mu_a:.4f} std={sd_a:.4f} rel_std={sd_a/mu_a*100 if mu_a else 0:.2f}%")
    print(f"coincidence_pct: mean={mu_c:.2f} std={sd_c:.2f} rel_std={sd_c/mu_c*100 if mu_c else 0:.2f}%")
