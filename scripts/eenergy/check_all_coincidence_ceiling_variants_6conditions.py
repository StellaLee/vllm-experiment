"""Compares all three coincidence-ceiling variants (DRF-family, weighted_sum, lmetric_power)
against their plain counterparts, across all 6 conditions where each has now been tested."""
from itertools import permutations
import compare_closedloopheavy_duration as c

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
        "lmetric_power", "lmetric_power_coincidence_ceiling", "round_robin"]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*95}\n{label}: {prefix}\n{'='*95}")
    vals = {}
    for arm in ARMS:
        trials = (4, 5, 6) if (prefix == "burstgptpergpu" and arm == "lmetric_power") else (1, 2, 3)
        try:
            vals[arm] = c.aggregate(prefix, arm, trials)
        except (FileNotFoundError, ZeroDivisionError) as e:
            print(f"  {arm}: MISSING/EMPTY")
            continue

    print(f"{'arm':<34}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'max_ramp':>12}{'ttft':>10}{'tbt':>10}")
    for arm in ARMS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<34}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
              f"{v['max_ramp'][0]:>12.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")

    scored = [a for a in ARMS if a != "round_robin" and a in vals]
    found = False
    print("Dominance:")
    for a, b in permutations(scored, 2):
        if c.dominates(vals[a], vals[b]):
            print(f"  {a} DOMINATES {b}")
            found = True
    if not found:
        print("  (none)")
