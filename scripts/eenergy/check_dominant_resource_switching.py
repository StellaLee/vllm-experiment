"""Tests the switching hypothesis directly: for drf_no_power, does the dominant resource
(compute vs load) for the CHOSEN candidate flip between consecutive decisions more often
than random chance would predict, given the observed base rate? This is the mechanism
hypothesis for why drf_no_power's p99_ramp is worse than either compute_only or load_only
alone despite a LOWER absolute max ramp -- a rule that keeps switching its own optimization
target plausibly produces more frequent moderate perturbations than one that commits to a
single consistent criterion throughout."""
import csv

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "closedloopheavylongpergpu"
ARM = "drf_no_power_diag"
TRIALS = (1, 2, 3)


def load_dominant_sequence(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            sc = row["share_compute"]
            sl = row["share_load"]
            if sc == "" or sl == "":
                continue
            sc, sl = float(sc), float(sl)
            if sc > sl:
                dom = "compute"
            elif sl > sc:
                dom = "load"
            else:
                dom = "tie"
            rows.append((float(row["wall_time"]), dom))
    rows.sort()
    return rows


all_doms = []
for trial in TRIALS:
    path = f"{LOGDIR}/{PREFIX}_assignment_{ARM}_t{trial}.csv"
    rows = load_dominant_sequence(path)
    all_doms.append([d for _, d in rows])

n_total = sum(len(d) for d in all_doms)
n_compute = sum(d.count("compute") for d in all_doms)
n_load = sum(d.count("load") for d in all_doms)
n_tie = sum(d.count("tie") for d in all_doms)

print(f"Total decisions: {n_total}")
print(f"  compute-dominant: {n_compute} ({n_compute/n_total:.1%})")
print(f"  load-dominant:    {n_load} ({n_load/n_total:.1%})")
print(f"  tied:             {n_tie} ({n_tie/n_total:.1%})")

# Switch rate: fraction of consecutive decision PAIRS (within each trial, temporal order)
# where the dominant resource differs from the previous decision.
n_pairs, n_switches = 0, 0
for d in all_doms:
    for a, b in zip(d, d[1:]):
        if a == "tie" or b == "tie":
            continue  # ties don't cleanly count as either resource "winning"
        n_pairs += 1
        if a != b:
            n_switches += 1

observed_rate = n_switches / n_pairs if n_pairs else 0

# Expected switch rate under an i.i.d. random-assignment null, given the observed base rate
# of compute vs load dominance: P(switch) = 2*p*(1-p) where p = P(compute).
p = n_compute / (n_compute + n_load) if (n_compute + n_load) else 0
expected_rate_null = 2 * p * (1 - p)

print(f"\nConsecutive-decision pairs: {n_pairs}")
print(f"Observed switch rate:  {observed_rate:.1%}")
print(f"Expected switch rate (i.i.d. null, same base rate): {expected_rate_null:.1%}")
print(f"Ratio (observed/expected): {observed_rate/expected_rate_null:.3f}" if expected_rate_null else "")
