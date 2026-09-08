from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
arms = {
    "drf_no_power": (1, 2, 3),
    "lmetric": (1, 2, 3),
    "coincidence_ceiling": (1, 2, 3),
    "lmetric_power": (1, 2, 3, 4, 5, 6),
    "drf_fixed": (1, 2, 3, 4, 5, 6),
}
vals = {a: c.aggregate(PREFIX, a, t) for a, t in arms.items()}

print(f"{'arm':<24}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'ttft':>10}{'tbt':>10}")
for a, v in vals.items():
    print(f"{a:<24}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
          f"{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")

print("\nDominance among all 5:")
found = False
for x, y in permutations(vals, 2):
    if c.dominates(vals[x], vals[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")
