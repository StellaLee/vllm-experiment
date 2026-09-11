import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
vals = {
    "coincidence_ceiling": c.aggregate(PREFIX, "coincidence_ceiling", (1, 2, 3, 4, 5, 6)),
    "lmetric_power_coincidence_ceiling": c.aggregate(PREFIX, "lmetric_power_coincidence_ceiling", (1, 2, 3, 4, 5, 6)),
}

print(f"{'arm':<38}{'mean_ramp':>16}{'p99_ramp':>16}{'max_ramp':>16}")
for arm, v in vals.items():
    print(f"{arm:<38}"
          f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<5.1f}"
          f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<5.1f}"
          f"{v['max_ramp'][0]:>10.1f}±{v['max_ramp'][1]:<5.1f}")

cc = vals["coincidence_ceiling"]
lp = vals["lmetric_power_coincidence_ceiling"]
print("\nDiff (lmetric_power_coincidence_ceiling - coincidence_ceiling), and z using pooled std:")
for m in ["mean_ramp", "p99_ramp", "max_ramp"]:
    diff = lp[m][0] - cc[m][0]
    pooled_sd = ((cc[m][1]**2 + lp[m][1]**2) / 2) ** 0.5
    z = diff / pooled_sd if pooled_sd else float("nan")
    print(f"  {m}: diff={diff:+.1f}, pooled_std={pooled_sd:.1f}, z={z:+.2f}")
