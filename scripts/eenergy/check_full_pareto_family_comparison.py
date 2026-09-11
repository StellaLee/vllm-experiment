from collections import defaultdict
from itertools import permutations
import compare_closedloopheavy_duration as c

CONDITIONS = {
    "Heavy/Closed-Loop (short)": "closedloopheavypergpu",
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
    "BurstGPT": "burstgptpergpu",
    "Ramp & Route": "rampandroutepergpu",
    "WildChat": "wildchatnaturalpergpu",
}

ARMS = [
    "round_robin",
    "lmetric",
    "compute_only",
    "load_only",
    "coincidence_ceiling",
    "weighted_sum_coincidence_ceiling",
    "lmetric_power_coincidence_ceiling",
    "lmetric_power_pareto_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_all_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_small_coincidence_ceiling",
]

SIX_TRIAL = {("closedloopheavylongpergpu", "round_robin"),
             ("closedloopheavylongpergpu", "compute_only"),
             ("closedloopheavylongpergpu", "load_only"),
             ("burstgptpergpu", "round_robin")}

dominance_tally = defaultdict(int)

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*105}\n{label}: {prefix}\n{'='*105}")
    vals = {}
    for arm in ARMS:
        trials = (1, 2, 3, 4, 5, 6) if (prefix, arm) in SIX_TRIAL else (1, 2, 3)
        try:
            vals[arm] = c.aggregate(prefix, arm, trials)
        except (FileNotFoundError, ZeroDivisionError):
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
            dominance_tally[x] += 1
            found = True
    if not found:
        print("  (none)")

print(f"\n{'='*60}\nOverall dominance tally (all 7 conditions)\n{'='*60}")
for arm, n in sorted(dominance_tally.items(), key=lambda kv: -kv[1]):
    print(f"  {n:>3}  {arm}")
