"""Partial check: compares BurstGPT's original 3-trial stats (trials 1-3) against the new
trials 4-6, for whichever arms have completed trials 4-6 so far -- lets us eyeball whether
the extra trials look consistent with the original before the rest of the battery finishes."""
import sys
sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import compare_closedloopheavy_duration as c

PREFIX = "burstgptpergpu"
ARMS_TO_CHECK = ["drf_fixed", "drf_power_tiebreak_full"]

for arm in ARMS_TO_CHECK:
    print(f"\n=== {arm} ===")
    for label, trials in [("trials 1-3 (original)", (1, 2, 3)),
                           ("trials 4-6 (new)", (4, 5, 6)),
                           ("trials 1-6 (combined)", (1, 2, 3, 4, 5, 6))]:
        try:
            v = c.aggregate(PREFIX, arm, trials)
        except FileNotFoundError as e:
            print(f"  {label}: MISSING ({e})")
            continue
        print(f"  {label}:")
        print(f"    peak={v['peak'][0]:.1f}±{v['peak'][1]:.1f}  "
              f"mean_ramp={v['mean_ramp'][0]:.1f}±{v['mean_ramp'][1]:.1f}  "
              f"p99_ramp={v['p99_ramp'][0]:.1f}±{v['p99_ramp'][1]:.1f}  "
              f"TTFT={v['ttft'][0]:.3f}±{v['ttft'][1]:.3f}  "
              f"TBT={v['tbt'][0]:.1f}±{v['tbt'][1]:.1f}")
