"""Early look at BurstGPT's full 5-arm comparison, using whatever trials are available so
far: drf_fixed/drf_power_tiebreak_full/weighted_sum get all 6 (1-3 original + 4-6 new);
lmetric_power gets 4-6 (brand new, never tested on BurstGPT before); round_robin gets
whatever's done out of 1-6 (t6 may still be in flight)."""
import sys
sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import compare_closedloopheavy_duration as c
from itertools import permutations

PREFIX = "burstgptpergpu"
TRIAL_SETS = {
    "drf_fixed": (1, 2, 3, 4, 5, 6),
    "drf_power_tiebreak_full": (1, 2, 3, 4, 5, 6),
    "weighted_sum": (1, 2, 3, 4, 5, 6),
    "lmetric_power": (4, 5, 6),
    "round_robin": (1, 2, 3, 4, 5),  # t6 may still be running
}

vals = {}
for arm, trials in TRIAL_SETS.items():
    try:
        vals[arm] = c.aggregate(PREFIX, arm, trials)
        print(f"{arm} (trials {trials}): OK")
    except FileNotFoundError as e:
        print(f"{arm} (trials {trials}): MISSING ({e})")

print(f"\n{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
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

print("\nDominance (scored arms):")
scored = [a for a in ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power"] if a in vals]
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
