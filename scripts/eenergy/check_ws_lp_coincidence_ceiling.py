"""Compares weighted_sum_coincidence_ceiling and lmetric_power_coincidence_ceiling against
their plain counterparts and drf_power_tiebreak_full_coincidence_ceiling on Heavy/Closed-
Loop's long condition."""
from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
ARM_TRIALS = {
    "weighted_sum": (1, 2, 3, 4, 5, 6),
    "weighted_sum_coincidence_ceiling": (1, 2, 3),
    "lmetric_power": (1, 2, 3, 4, 5, 6),
    "lmetric_power_coincidence_ceiling": (1, 2, 3),
    "coincidence_ceiling": (1, 2, 3),
    "drf_power_tiebreak_full": (1, 2, 3, 4, 5, 6),
}

vals = {}
for arm, trials in ARM_TRIALS.items():
    try:
        vals[arm] = c.aggregate(PREFIX, arm, trials)
    except (FileNotFoundError, ZeroDivisionError) as e:
        print(f"{arm}: MISSING/EMPTY ({e})")

print(f"{'arm':<34}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'ttft':>10}{'tbt':>10}")
for arm in ARM_TRIALS:
    if arm not in vals:
        continue
    v = vals[arm]
    print(f"{arm:<34}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
          f"{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")

print("\nDominance:")
found = False
for a, b in permutations(vals, 2):
    if c.dominates(vals[a], vals[b]):
        print(f"  {a} DOMINATES {b}")
        found = True
if not found:
    print("  (none)")
