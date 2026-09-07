"""Same as compare_closedloopheavy_duration.py but explicitly compares SHORT (3 trials) vs.
LONG (all 6 trials, 1-6) now that trials 4-6 have completed."""
from itertools import permutations
import compare_closedloopheavy_duration as c

for label, prefix, trials in [
    ("SHORT (~51-53s), 3 trials", "closedloopheavypergpu", (1, 2, 3)),
    ("LONG (~225-232s), 6 trials", "closedloopheavylongpergpu", (1, 2, 3, 4, 5, 6)),
]:
    print(f"\n{'='*90}\n{label}: {prefix}\n{'='*90}")
    vals = {arm: c.aggregate(prefix, arm, trials) for arm in c.ARMS}
    print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
    for arm in c.ARMS:
        v = vals[arm]
        print(f"{arm:<26}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
              f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    print("\nDominance (scored arms):")
    scored = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power"]
    found = False
    for a, b in permutations(scored, 2):
        if c.dominates(vals[a], vals[b]):
            print(f"  {a} DOMINATES {b}")
            found = True
    if not found:
        print("  (none)")

    print("round_robin vs each scored arm:")
    for arm in scored:
        if c.dominates(vals["round_robin"], vals[arm]):
            print(f"  round_robin DOMINATES {arm}")
        elif c.dominates(vals[arm], vals["round_robin"]):
            print(f"  {arm} DOMINATES round_robin")
        else:
            print(f"  round_robin vs {arm}: incomparable")
