from itertools import permutations
import compare_closedloopheavy_duration as c

CONDITIONS = {
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
}
# arm -> (available trial tuple per condition, or None to use default)
ARMS = ["round_robin", "lmetric", "compute_only", "load_only",
        "coincidence_ceiling", "lmetric_power_pareto_epsilon_coincidence_ceiling",
        "lmetric_power_pareto_epsilon_small_coincidence_ceiling"]

SIX_TRIAL_ARMS = {"compute_only", "load_only", "round_robin"}

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*100}\n{label}: {prefix}\n{'='*100}")
    vals = {}
    for arm in ARMS:
        if prefix == "closedloopheavylongpergpu" and arm in SIX_TRIAL_ARMS:
            trials = (1, 2, 3, 4, 5, 6)
        else:
            trials = (1, 2, 3)
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

    # Rank on the two tail metrics specifically
    print("\nRanking on p99_ramp (lower=better):")
    for arm, v in sorted(vals.items(), key=lambda kv: kv[1]['p99_ramp'][0]):
        print(f"  {v['p99_ramp'][0]:>10.1f}  {arm}")
    print("Ranking on max_ramp (lower=better):")
    for arm, v in sorted(vals.items(), key=lambda kv: kv[1]['max_ramp'][0]):
        print(f"  {v['max_ramp'][0]:>10.1f}  {arm}")
