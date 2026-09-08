"""lmetric (power-blind) now complete on all 6 conditions -- compares against lmetric_power
per condition."""
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

for label, prefix in CONDITIONS.items():
    print(f"\n{'='*70}\n{label}\n{'='*70}")
    lp_trials = (4, 5, 6) if prefix == "burstgptpergpu" else (1, 2, 3)
    try:
        lp = c.aggregate(prefix, "lmetric_power", lp_trials)
        lm = c.aggregate(prefix, "lmetric", (1, 2, 3))
    except (FileNotFoundError, ZeroDivisionError) as e:
        print(f"  MISSING/EMPTY: {e}")
        continue
    print(f"{'arm':<16}{'peak':>10}{'mean_ramp':>12}{'p99_ramp':>12}{'ttft':>10}{'tbt':>10}")
    for name, v in [("lmetric_power", lp), ("lmetric", lm)]:
        print(f"{name:<16}{v['peak'][0]:>10.1f}{v['mean_ramp'][0]:>12.1f}{v['p99_ramp'][0]:>12.1f}"
              f"{v['ttft'][0]:>10.3f}{v['tbt'][0]:>10.1f}")
    if c.dominates(lp, lm):
        print("  lmetric_power DOMINATES lmetric")
    elif c.dominates(lm, lp):
        print("  lmetric DOMINATES lmetric_power")
    else:
        print("  incomparable")
