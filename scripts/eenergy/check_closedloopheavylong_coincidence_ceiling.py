"""Compares coincidence_ceiling against the existing 5 arms on Heavy/Closed-Loop's long
(~225-232s) condition, 6-trial baseline for the existing arms vs. 3 trials for the new one
(consistent with how lmetric_power_pareto was first checked)."""
from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
TRIAL_SETS = {
    "drf_fixed": (1, 2, 3, 4, 5, 6),
    "drf_power_tiebreak_full": (1, 2, 3, 4, 5, 6),
    "weighted_sum": (1, 2, 3, 4, 5, 6),
    "lmetric_power": (1, 2, 3, 4, 5, 6),
    "round_robin": (1, 2, 3, 4, 5, 6),
    "coincidence_ceiling": (1, 2, 3),
}

vals = {}
for arm, trials in TRIAL_SETS.items():
    try:
        vals[arm] = c.aggregate(PREFIX, arm, trials)
    except FileNotFoundError as e:
        print(f"{arm}: MISSING ({e})")

print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
for arm in TRIAL_SETS:
    if arm not in vals:
        continue
    v = vals[arm]
    print(f"{arm:<26}"
          f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
          f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
          f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
          f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
          f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

print("\nDominance (all scored arms, coincidence_ceiling included):")
scored = [a for a in ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum",
                       "lmetric_power", "coincidence_ceiling"] if a in vals]
found = False
for a, b in permutations(scored, 2):
    if c.dominates(vals[a], vals[b]):
        print(f"  {a} DOMINATES {b}")
        found = True
if not found:
    print("  (none)")

if "round_robin" in vals:
    print("round_robin vs each scored arm:")
    for arm in scored:
        if c.dominates(vals["round_robin"], vals[arm]):
            print(f"  round_robin DOMINATES {arm}")
        elif c.dominates(vals[arm], vals["round_robin"]):
            print(f"  {arm} DOMINATES round_robin")
        else:
            print(f"  round_robin vs {arm}: incomparable")
