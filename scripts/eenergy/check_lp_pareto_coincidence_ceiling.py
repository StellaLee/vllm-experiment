from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
ARM_TRIALS = {
    "drf_power_tiebreak_full": (1, 2, 3, 4, 5, 6),
    "coincidence_ceiling": (1, 2, 3),
    "weighted_sum_coincidence_ceiling": (1, 2, 3),
    "lmetric_power_coincidence_ceiling": (1, 2, 3),
    "lmetric_power_pareto_coincidence_ceiling": (1, 2, 3),
}
vals = {}
for a, t in ARM_TRIALS.items():
    try:
        vals[a] = c.aggregate(PREFIX, a, t)
    except (FileNotFoundError, ZeroDivisionError):
        print(f"{a}: MISSING/EMPTY on this condition")

print(f"{'arm':<40}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'max_ramp':>12}{'ttft':>10}{'tbt':>10}")
for a, v in vals.items():
    print(f"{a:<40}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
          f"{v['max_ramp'][0]:>12.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")

print("\nDominance:")
found = False
for x, y in permutations(vals, 2):
    if c.dominates(vals[x], vals[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")
