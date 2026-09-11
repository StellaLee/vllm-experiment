from itertools import permutations
import compare_closedloopheavy_duration as c

print("=" * 100)
print("Heavy/Closed-Loop LONG: all Pareto-safe lmetric_power variants vs plain/unsafe versions")
print("=" * 100)
PREFIX1 = "closedloopheavylongpergpu"
ARM_TRIALS_1 = {
    "drf_power_tiebreak_full": (1, 2, 3, 4, 5, 6),
    "coincidence_ceiling": (1, 2, 3),
    "lmetric_power_coincidence_ceiling": (1, 2, 3),
    "lmetric_power_pareto_coincidence_ceiling": (1, 2, 3),
    "lmetric_power_pareto_epsilon_coincidence_ceiling": (1, 2, 3),
    "lmetric_power_pareto_epsilon_all_coincidence_ceiling": (1, 2, 3),
}
vals1 = {a: c.aggregate(PREFIX1, a, t) for a, t in ARM_TRIALS_1.items()}
print(f"{'arm':<52}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'ttft':>10}{'tbt':>10}")
for a, v in vals1.items():
    print(f"{a:<52}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
          f"{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")
print("\nDominance:")
found = False
for x, y in permutations(vals1, 2):
    if c.dominates(vals1[x], vals1[y]):
        print(" ", x, "DOMINATES", y)
        found = True
if not found:
    print("  (none)")

print()
print("=" * 100)
print("Dominance tally across all 6 remaining conditions (BurstGPT, Heavy/Matched, Cachehit,")
print("Heavy/Closed-Loop short, Ramp & Route, WildChat) -- all Pareto-safe lmetric_power variants")
print("=" * 100)
CONDITIONS = {
    "BurstGPT": "burstgptpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
    "Heavy/Closed-Loop (short)": "closedloopheavypergpu",
    "Ramp & Route": "rampandroutepergpu",
    "WildChat": "wildchatnaturalpergpu",
}
ARMS = ["drf_fixed", "drf_power_tiebreak_full", "coincidence_ceiling",
        "weighted_sum", "weighted_sum_coincidence_ceiling",
        "lmetric_power", "lmetric_power_coincidence_ceiling",
        "lmetric_power_pareto_coincidence_ceiling",
        "lmetric_power_pareto_epsilon_coincidence_ceiling",
        "lmetric_power_pareto_epsilon_all_coincidence_ceiling"]

tally = {a: 0 for a in ARMS}
for label, prefix in CONDITIONS.items():
    print(f"\n--- {label} ---")
    vals = {}
    for arm in ARMS:
        trials = (4, 5, 6) if (prefix == "burstgptpergpu" and arm == "lmetric_power") else (1, 2, 3)
        try:
            vals[arm] = c.aggregate(prefix, arm, trials)
        except (FileNotFoundError, ZeroDivisionError):
            continue
    found = False
    for x, y in permutations(vals, 2):
        if c.dominates(vals[x], vals[y]):
            print(f"  {x} DOMINATES {y}")
            tally[x] = tally.get(x, 0) + 1
            found = True
    if not found:
        print("  (none)")

print("\n" + "=" * 100)
print("FULL TALLY (dominance wins, all 7 conditions combined)")
print("=" * 100)
for a, n in sorted(tally.items(), key=lambda kv: -kv[1]):
    print(f"  {a:<52}{n}")
