from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "cachehitpergpu"
ARM_TRIALS = {
    "drf_fixed": (1, 2, 3),
    "drf_power_tiebreak_full": (1, 2, 3),
    "weighted_sum": (1, 2, 3),
    "lmetric_power": (1, 2, 3),
    "lmetric": (1, 2, 3),
    "coincidence_ceiling": (1, 2, 3),
    "drf_no_power": (1, 2, 3),
    "compute_only": (1, 2, 3),
    "load_only": (1, 2, 3),
    "round_robin": (1, 2, 3),
}
vals = {a: c.aggregate(PREFIX, a, t) for a, t in ARM_TRIALS.items()}

print(f"{'arm':<26}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'ttft':>10}{'tbt':>10}")
for a, v in vals.items():
    print(f"{a:<26}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
          f"{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")

scored = [a for a in ARM_TRIALS if a != "round_robin"]
print("\nDominance:")
found = False
for x, y in permutations(scored, 2):
    if c.dominates(vals[x], vals[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")

print("\nTTFT ranking (best to worst):")
for a in sorted(vals, key=lambda a: vals[a]['ttft'][0]):
    print(f"  {a:<26}{vals[a]['ttft'][0]:.3f}")
