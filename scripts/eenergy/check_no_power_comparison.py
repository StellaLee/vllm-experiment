"""Compares power-blind arms (drf_no_power, weighted_sum_no_power, lmetric) against their
power-aware siblings on Heavy/Closed-Loop's long condition -- does power-awareness in the
routing decision actually help the power/ramp outcomes it's meant to protect?"""
from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
PAIRS = [
    ("drf_fixed", "drf_no_power"),
    ("weighted_sum", "weighted_sum_no_power"),
    ("lmetric_power", "lmetric"),
]
TRIALS_EXISTING = (1, 2, 3, 4, 5, 6)
TRIALS_NEW = (1, 2, 3)

vals = {}
for power_arm, blind_arm in PAIRS:
    for arm, trials in [(power_arm, TRIALS_EXISTING), (blind_arm, TRIALS_NEW)]:
        try:
            vals[arm] = c.aggregate(PREFIX, arm, trials)
        except FileNotFoundError as e:
            print(f"{arm}: MISSING ({e})")

print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
for power_arm, blind_arm in PAIRS:
    for arm in (power_arm, blind_arm):
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<26}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
              f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")
    print()

print("Head-to-head, power-aware vs power-blind:")
for power_arm, blind_arm in PAIRS:
    if power_arm not in vals or blind_arm not in vals:
        continue
    if c.dominates(vals[power_arm], vals[blind_arm]):
        print(f"  {power_arm} DOMINATES {blind_arm}")
    elif c.dominates(vals[blind_arm], vals[power_arm]):
        print(f"  {blind_arm} DOMINATES {power_arm}")
    else:
        print(f"  {power_arm} vs {blind_arm}: incomparable")
