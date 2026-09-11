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

# arm -> (label, epsilon value for reference)
ARMS = [
    ("lmetric_power_coincidence_ceiling", "unsafe original (eps->0 limit)"),
    ("lmetric_power_pareto_epsilon_small_coincidence_ceiling", "eps=0.001"),
    ("lmetric_power_pareto_epsilon_coincidence_ceiling", "eps=0.01"),
    ("lmetric_power_pareto_coincidence_ceiling", "eps=1"),
]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*100}\n{label}: {prefix}\n{'='*100}")
    vals = {}
    for arm, tag in ARMS:
        try:
            vals[arm] = c.aggregate(prefix, arm, (1, 2, 3))
        except (FileNotFoundError, ZeroDivisionError):
            continue
    print(f"{'arm':<58}{'peak':>16}{'mean_ramp':>16}{'p99_ramp':>16}{'max_ramp':>16}{'ttft':>12}{'tbt':>14}")
    for arm, tag in ARMS:
        if arm not in vals:
            continue
        v = vals[arm]
        print(f"{arm+' ('+tag+')':<58}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<5.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<5.1f}"
              f"{v['max_ramp'][0]:>10.1f}±{v['max_ramp'][1]:<5.1f}"
              f"{v['ttft'][0]:>7.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    if "lmetric_power_coincidence_ceiling" in vals:
        base = vals["lmetric_power_coincidence_ceiling"]
        print("\n  abs distance from lmetric_power_coincidence_ceiling (lower = closer):")
        for arm, tag in ARMS[1:]:
            if arm not in vals:
                continue
            v = vals[arm]
            dists = {k: abs(v[k][0] - base[k][0]) for k in ["mean_ramp", "p99_ramp", "max_ramp"]}
            print(f"    {tag:<12} mean_ramp={dists['mean_ramp']:.1f}  "
                  f"p99_ramp={dists['p99_ramp']:.1f}  max_ramp={dists['max_ramp']:.1f}")
