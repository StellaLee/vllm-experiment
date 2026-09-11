"""Side-by-side peak/mean_ramp/p99_ramp/max_ramp/ttft/tbt comparison of
drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp vs coincidence_ceiling, across all 7
conditions (6 expansion conditions + the long-condition reference). Reuses the same aggregate()
helper (fleet-level ramp stats from the power_trace sidecar, consistent methodology across
every arm this session) already used for every other arm comparison."""
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

ARMS = ["coincidence_ceiling", "drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp"]

for label, prefix in CONDITIONS.items():
    trials = (1, 2, 3) if prefix != "closedloopheavylongpergpu" else (1, 2, 3, 4, 5, 6)
    print(f"\n{'='*100}\n{label}: {prefix}\n{'='*100}")
    vals = {}
    for arm in ARMS:
        use_trials = trials
        try:
            vals[arm] = c.aggregate(prefix, arm, use_trials)
        except (FileNotFoundError, ZeroDivisionError):
            use_trials = (1, 2, 3)
            try:
                vals[arm] = c.aggregate(prefix, arm, use_trials)
            except (FileNotFoundError, ZeroDivisionError):
                print(f"  {arm}: MISSING/EMPTY")
                continue

    print(f"{'arm':<62}{'peak':>10}{'mean_ramp':>13}{'p99_ramp':>13}{'max_ramp':>13}{'ttft':>10}{'tbt':>10}")
    for arm in ARMS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm:<62}{v['peak'][0]:>7.1f}±{v['peak'][1]:<4.1f}"
              f"{v['mean_ramp'][0]:>9.1f}±{v['mean_ramp'][1]:<4.1f}"
              f"{v['p99_ramp'][0]:>9.1f}±{v['p99_ramp'][1]:<4.1f}"
              f"{v['max_ramp'][0]:>9.1f}±{v['max_ramp'][1]:<4.1f}"
              f"{v['ttft'][0]:>7.3f}±{v['ttft'][1]:<3.3f}"
              f"{v['tbt'][0]:>7.1f}±{v['tbt'][1]:<3.1f}")
