from itertools import permutations
import compare_closedloopheavy_duration as c

CONDITIONS = {
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
}
ARMS = ["round_robin", "lmetric", "compute_only", "load_only", "lmetric_power_pareto_epsilon_coincidence_ceiling"]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*90}\n{label}: {prefix}\n{'='*90}")
    vals = {}
    for arm in ARMS:
        trials = (1, 2, 3, 4, 5, 6) if (prefix == "closedloopheavylongpergpu" and arm in ("compute_only", "load_only", "round_robin")) else (1, 2, 3)
        try:
            vals[arm] = c.aggregate(prefix, arm, trials)
        except (FileNotFoundError, ZeroDivisionError) as e:
            print(f"  {arm}: MISSING/EMPTY")
            continue
    print(f"{'arm':<52}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'max_ramp':>12}{'ttft':>10}{'tbt':>10}")
    for arm in ARMS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<52}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
              f"{v['max_ramp'][0]:>12.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")
    print("\nDominance:")
    found = False
    for x, y in permutations(vals, 2):
        if c.dominates(vals[x], vals[y]):
            print(f"  {x} DOMINATES {y}")
            found = True
    if not found:
        print("  (none)")
