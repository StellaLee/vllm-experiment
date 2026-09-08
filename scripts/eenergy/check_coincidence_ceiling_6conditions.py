"""Compares coincidence_ceiling against the existing 4 scored arms + round_robin across all
6 conditions where it was just validated: BurstGPT, Heavy/Matched, Light/Cachehit,
Heavy/Closed-Loop (short), Ramp & Route, WildChat."""
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
ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power",
        "coincidence_ceiling", "round_robin"]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*90}\n{label}: {prefix}\n{'='*90}")
    vals = {}
    for arm in ARMS:
        try:
            vals[arm] = c.aggregate(prefix, arm, (1, 2, 3))
        except FileNotFoundError as e:
            print(f"  {arm}: MISSING ({e})")

    print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
    for arm in ARMS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<26}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
              f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    scored = [a for a in ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum",
                           "lmetric_power", "coincidence_ceiling"] if a in vals]
    found = False
    print("Dominance:")
    for a, b in permutations(scored, 2):
        if c.dominates(vals[a], vals[b]):
            print(f"  {a} DOMINATES {b}")
            found = True
    if not found:
        print("  (none)")
