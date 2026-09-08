from itertools import permutations
import compare_closedloopheavy_duration as c

print("=" * 70)
print("Heavy/Closed-Loop LONG: compute_only/load_only at n=6")
print("=" * 70)
PREFIX1 = "closedloopheavylongpergpu"
ARM_TRIALS_1 = {
    "drf_fixed": (1, 2, 3, 4, 5, 6),
    "drf_no_power": (1, 2, 3),
    "compute_only": (1, 2, 3, 4, 5, 6),
    "load_only": (1, 2, 3, 4, 5, 6),
    "coincidence_ceiling": (1, 2, 3),
}
vals1 = {a: c.aggregate(PREFIX1, a, t) for a, t in ARM_TRIALS_1.items()}
print(f"{'arm':<20}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>14}{'ttft':>10}{'tbt':>10}")
for a, v in vals1.items():
    print(f"{a:<20}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}"
          f"{v['p99_ramp'][0]:>9.1f}±{v['p99_ramp'][1]:<4.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")
print("\nDominance:")
found = False
for x, y in permutations(vals1, 2):
    if c.dominates(vals1[x], vals1[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")

print()
print("=" * 70)
print("Heavy/Matched: compute_only/load_only replication check (n=3)")
print("=" * 70)
PREFIX2 = "openloopwhalelongoutmatchedpergpu"
ARM_TRIALS_2 = {
    "drf_fixed": (1, 2, 3),
    "drf_power_tiebreak_full": (1, 2, 3),
    "lmetric_power": (1, 2, 3),
    "coincidence_ceiling": (1, 2, 3),
    "drf_no_power": (1, 2, 3),
    "compute_only": (1, 2, 3),
    "load_only": (1, 2, 3),
}
vals2 = {}
for a, t in ARM_TRIALS_2.items():
    try:
        vals2[a] = c.aggregate(PREFIX2, a, t)
    except (FileNotFoundError, ZeroDivisionError) as e:
        print(f"{a}: MISSING/EMPTY ({e})")
print(f"{'arm':<26}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>14}{'ttft':>10}{'tbt':>10}")
for a, v in vals2.items():
    print(f"{a:<26}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}"
          f"{v['p99_ramp'][0]:>9.1f}±{v['p99_ramp'][1]:<4.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")
print("\nDominance:")
found = False
for x, y in permutations(vals2, 2):
    if c.dominates(vals2[x], vals2[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")
