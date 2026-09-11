import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
ARMS = [
    "lmetric_power_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_zero_coincidence_ceiling",
    "lmetric_power_pareto_epsilon_coincidence_ceiling",
    "lmetric_power_pareto_coincidence_ceiling",
]

vals = {}
for arm in ARMS:
    vals[arm] = c.aggregate(PREFIX, arm, (1, 2, 3))

print(f"{'arm':<58}{'peak':>16}{'mean_ramp':>16}{'p99_ramp':>16}{'max_ramp':>16}{'ttft':>12}{'tbt':>14}")
for arm in ARMS:
    v = vals[arm]
    print(f"{arm:<58}"
          f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
          f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<5.1f}"
          f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<5.1f}"
          f"{v['max_ramp'][0]:>10.1f}±{v['max_ramp'][1]:<5.1f}"
          f"{v['ttft'][0]:>7.3f}±{v['ttft'][1]:<5.3f}"
          f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")

print("\n% difference of epsilon_zero from lmetric_power_coincidence_ceiling (the predicted match):")
base = vals["lmetric_power_coincidence_ceiling"]
ez = vals["lmetric_power_pareto_epsilon_zero_coincidence_ceiling"]
for k in ["peak", "mean_ramp", "p99_ramp", "max_ramp", "ttft", "tbt"]:
    pct = (ez[k][0] - base[k][0]) / base[k][0] * 100
    print(f"  {k}: {pct:+.1f}%")

print("\n% difference of epsilon_zero from lmetric_power_pareto (the epsilon=1 case, NOT predicted to match):")
p1 = vals["lmetric_power_pareto_coincidence_ceiling"]
for k in ["peak", "mean_ramp", "p99_ramp", "max_ramp", "ttft", "tbt"]:
    pct = (ez[k][0] - p1[k][0]) / p1[k][0] * 100
    print(f"  {k}: {pct:+.1f}%")
