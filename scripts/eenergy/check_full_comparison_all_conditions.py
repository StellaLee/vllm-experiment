"""Full comparison: every arm tested so far, every condition tested so far. Tries the
largest available trial set per arm (1-6, then 1-3, then 4-6 for BurstGPT's lmetric_power
special case) and reports missing/empty gracefully rather than crashing."""
import compare_closedloopheavy_duration as c

CONDITIONS = {
    "Heavy/Closed-Loop (short, ~51-53s)": "closedloopheavypergpu",
    "Heavy/Closed-Loop (long, ~225-232s)": "closedloopheavylongpergpu",
    "BurstGPT": "burstgptpergpu",
    "Heavy/Matched": "openloopwhalelongoutmatchedpergpu",
    "Light/Cachehit": "cachehitpergpu",
    "Ramp & Route": "rampandroutepergpu",
    "WildChat": "wildchatnaturalpergpu",
}

ARMS = ["drf_fixed", "drf_power_tiebreak_full", "weighted_sum", "lmetric_power",
        "lmetric", "coincidence_ceiling", "weighted_sum_coincidence_ceiling",
        "lmetric_power_coincidence_ceiling", "lmetric_power_pareto",
        "drf_no_power", "compute_only", "load_only", "round_robin"]

TRIAL_CANDIDATES = [(1, 2, 3, 4, 5, 6), (1, 2, 3), (4, 5, 6)]


def try_aggregate(prefix, arm):
    for trials in TRIAL_CANDIDATES:
        try:
            return c.aggregate(prefix, arm, trials), trials
        except FileNotFoundError:
            continue
        except ZeroDivisionError:
            return "EMPTY", None
    return None, None


for label, prefix in CONDITIONS.items():
    print(f"\n{'='*95}\n{label}: {prefix}\n{'='*95}")
    print(f"{'arm':<34}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'max_ramp':>12}{'ttft':>10}{'tbt':>10}  trials")
    for arm in ARMS:
        result, trials = try_aggregate(prefix, arm)
        if result is None:
            continue  # not tested on this condition, skip silently
        if result == "EMPTY":
            print(f"{arm:<34}  EMPTY (records file has zero valid entries)")
            continue
        v = result
        print(f"{arm:<34}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
              f"{v['max_ramp'][0]:>12.1f}{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}  n={len(trials)}")
