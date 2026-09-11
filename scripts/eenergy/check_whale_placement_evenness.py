"""Checks whether compute_only/load_only's p99_ramp advantage over drf_no_power is explained
by how EVENLY whale requests get spread across the 6 replicas -- more even spread avoids
correlated/coincident ramp events. Classifies whale vs non-whale via raw_tokens > 4000
(matching router_core.py's WHALE_TOKEN_THRESHOLD), then checks each arm's whale-placement
distribution across replicas."""
import csv
from collections import Counter

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "closedloopheavylongpergpu"
WHALE_TOKEN_THRESHOLD = 4000
ARMS = ["round_robin", "compute_only", "load_only", "drf_no_power"]


def whale_placement_counts(prefix, arm, trials):
    counts = Counter()
    n_whales = 0
    for trial in trials:
        path = f"{LOGDIR}/{prefix}_assignment_{arm}_t{trial}.csv"
        with open(path) as f:
            r = csv.DictReader(f)
            for row in r:
                raw = int(row["raw_tokens"])
                if raw > WHALE_TOKEN_THRESHOLD:
                    counts[row["gpu_index"]] += 1
                    n_whales += 1
    return counts, n_whales


TRIALS = {
    "round_robin": (1, 2, 3, 4, 5, 6),
    "compute_only": (1, 2, 3, 4, 5, 6),
    "load_only": (1, 2, 3, 4, 5, 6),
    "drf_no_power": (1, 2, 3),
}

for arm in ARMS:
    counts, n_whales = whale_placement_counts(PREFIX, arm, TRIALS[arm])
    vals = sorted(counts.values())
    if not vals:
        print(f"{arm}: no whales found")
        continue
    mean_v = sum(vals) / len(vals)
    max_v, min_v = max(vals), min(vals)
    # coefficient of variation as an evenness measure
    var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
    cv = (var ** 0.5) / mean_v if mean_v else 0
    print(f"{arm:<16} n_whales={n_whales:4d}  per-GPU counts={dict(sorted(counts.items()))}  "
          f"mean={mean_v:.1f} min={min_v} max={max_v}  CV={cv:.3f}")
