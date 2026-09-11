import compare_closedloopheavy_duration as c

PREFIX = "closedloopheavylongpergpu"
ARM = "lmetric_power_coincidence_ceiling"

v3 = c.aggregate(PREFIX, ARM, (1, 2, 3))
v6 = c.aggregate(PREFIX, ARM, (1, 2, 3, 4, 5, 6))
v456 = c.aggregate(PREFIX, ARM, (4, 5, 6))

print(f"{'trials':<10}{'peak':>16}{'mean_ramp':>16}{'p99_ramp':>16}{'max_ramp':>16}{'ttft':>12}{'tbt':>14}")
for label, v in [("1-3", v3), ("4-6", v456), ("1-6", v6)]:
    print(f"{label:<10}"
          f"{v['peak'][0]:>10.1f}±{v['peak'][1]:<5.1f}"
          f"{v['mean_ramp'][0]:>10.1f}±{v['mean_ramp'][1]:<5.1f}"
          f"{v['p99_ramp'][0]:>10.1f}±{v['p99_ramp'][1]:<5.1f}"
          f"{v['max_ramp'][0]:>10.1f}±{v['max_ramp'][1]:<5.1f}"
          f"{v['ttft'][0]:>7.3f}±{v['ttft'][1]:<5.3f}"
          f"{v['tbt'][0]:>8.1f}±{v['tbt'][1]:<5.1f}")
