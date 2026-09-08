"""Early reading: drf_no_power (now complete on all 6 conditions) against every other arm,
including coincidence_ceiling, per condition."""
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
ARM_TRIALS = {
    "drf_fixed": (1, 2, 3),
    "drf_power_tiebreak_full": (1, 2, 3),
    "weighted_sum": (1, 2, 3),
    "lmetric_power": (1, 2, 3),
    "coincidence_ceiling": (1, 2, 3),
    "drf_no_power": (1, 2, 3),
    "round_robin": (1, 2, 3),
}

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*90}\n{label}: {prefix}\n{'='*90}")
    vals = {}
    for arm, trials in ARM_TRIALS.items():
        use_trials = trials
        if prefix == "burstgptpergpu" and arm == "lmetric_power":
            use_trials = (4, 5, 6)  # BurstGPT's lmetric_power was only ever collected as t4-6
        try:
            vals[arm] = c.aggregate(prefix, arm, use_trials)
        except FileNotFoundError as e:
            print(f"  {arm}: MISSING ({e})")
        except ZeroDivisionError:
            print(f"  {arm}: EMPTY (one or more trial files had zero valid records)")

    print(f"{'arm':<26}{'peak(W)':>16}{'mean_ramp(W/s)':>18}{'p99_ramp(W/s)':>18}{'TTFT(s)':>14}{'TBT(ms)':>14}")
    for arm in ARM_TRIALS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<26}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<7.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<7.1f}"
              f"{v['ttft'][0]:>8.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    scored = [a for a in ARM_TRIALS if a != "round_robin" and a in vals]
    found = False
    print("Dominance:")
    for a, b in permutations(scored, 2):
        if c.dominates(vals[a], vals[b]):
            print(f"  {a} DOMINATES {b}")
            found = True
    if not found:
        print("  (none)")
