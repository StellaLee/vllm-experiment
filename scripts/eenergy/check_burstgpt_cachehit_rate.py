"""Computes the actual cache-hit rate for BurstGPT dispatches, using the assignment logs'
new_tokens/raw_tokens fields (new_tokens = P-token, the prefix-cache-discounted count of NEW
prefill tokens the chosen replica would need; raw_tokens = the request's actual prompt length).
A full cache hit is new_tokens == 0. Also reports the mean new/raw ratio (a continuous
"average miss fraction") since partial hits matter too, not just exact zeros."""
import csv
import glob
import os

LOGDIR = "/root/pli/vllm-experiment/logs"
ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power", "round_robin"]


def analyze(prefix, arm, trials):
    n_total = 0
    n_full_hit = 0
    ratios = []
    for trial in trials:
        path = f"{LOGDIR}/{prefix}_assignment_{arm}_t{trial}.csv"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                raw = int(row["raw_tokens"])
                new = int(row["new_tokens"])
                if raw <= 1:
                    continue  # skip degenerate warmup/probe rows
                n_total += 1
                if new == 0:
                    n_full_hit += 1
                ratios.append(new / raw)
    if n_total == 0:
        return None
    mean_ratio = sum(ratios) / len(ratios)
    return n_total, n_full_hit, n_full_hit / n_total, mean_ratio


TRIAL_SETS = {
    "drf_fixed": range(1, 7),
    "drf_power_tiebreak_full": range(1, 7),
    "weighted_sum": range(1, 7),
    "lmetric_power": range(4, 7),
    "round_robin": range(1, 6),
}

print(f"{'arm':<26}{'n':>8}{'full_hits':>12}{'hit_rate':>12}{'mean_new/raw':>16}")
for arm in ARMS:
    res = analyze("burstgptpergpu", arm, TRIAL_SETS[arm])
    if res is None:
        print(f"{arm:<26} no data")
        continue
    n_total, n_full_hit, hit_rate, mean_ratio = res
    print(f"{arm:<26}{n_total:>8}{n_full_hit:>12}{hit_rate:>11.1%}{mean_ratio:>15.3f}")
