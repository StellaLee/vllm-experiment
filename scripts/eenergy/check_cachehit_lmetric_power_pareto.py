"""Compares lmetric_power_pareto against the existing 5 arms on Light/Cachehit's original
condition -- the highest-cache-hit-rate condition, so the one most likely to reveal whether
the Pareto-safety fix changes empirical behavior vs. plain lmetric_power."""
from itertools import permutations
import compare_closedloopheavy_duration as c

PREFIX = "cachehitpergpu"
ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power",
        "lmetric_power_pareto", "round_robin"]

vals = {}
for arm in ARMS:
    try:
        vals[arm] = c.aggregate(PREFIX, arm, (1, 2, 3))
    except FileNotFoundError as e:
        print(f"{arm}: MISSING ({e})")

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

print("\nDominance (all scored arms):")
scored = [a for a in ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum",
                       "lmetric_power", "lmetric_power_pareto"] if a in vals]
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
