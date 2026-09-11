import compare_closedloopheavy_duration as c

CONDITIONS = {
    "Heavy/Closed-Loop (short)": "closedloopheavypergpu",
    "Heavy/Closed-Loop (long)": "closedloopheavylongpergpu",
}
ARMS = ["coincidence_ceiling", "lmetric", "round_robin"]

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*100}\n{label}: {prefix} (n=6 each)\n{'='*100}")
    vals = {}
    for arm in ARMS:
        vals[arm] = c.aggregate(prefix, arm, (1, 2, 3, 4, 5, 6))
    print(f"{'arm':<24}{'peak':>16}{'mean_ramp':>16}{'p99_ramp':>16}{'max_ramp':>16}{'ttft':>12}{'tbt':>14}")
    for arm in ARMS:
        v = vals[arm]
        print(f"{arm:<24}"
              f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
              f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<5.1f}"
              f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<5.1f}"
              f"{v['max_ramp'][0]:>10.1f}±{v['max_ramp'][1]:<5.1f}"
              f"{v['ttft'][0]:>7.3f}±{v['ttft'][1]:<5.3f}"
              f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

    print("\n  z-score of coincidence_ceiling vs round_robin and lmetric (using coincidence_ceiling's own std as scale):")
    cc = vals["coincidence_ceiling"]
    for other in ["round_robin", "lmetric"]:
        ov = vals[other]
        print(f"  vs {other}:")
        for m in ["mean_ramp", "p99_ramp", "max_ramp"]:
            diff = cc[m][0] - ov[m][0]
            scale = cc[m][1] if cc[m][1] > 0 else 1e-9
            z = diff / scale
            print(f"    {m}: coincidence_ceiling={cc[m][0]:.1f} vs {other}={ov[m][0]:.1f}, "
                  f"diff={diff:+.1f}, z={z:+.2f}")

    print("\n  Dominance:")
    from itertools import permutations
    found = False
    for x, y in permutations(vals, 2):
        if c.dominates(vals[x], vals[y]):
            print(f"    {x} DOMINATES {y}")
            found = True
    if not found:
        print("    (none)")
