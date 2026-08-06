#!/usr/bin/env python3
"""Diagnose WHY peak power didn't differ between mono and chunk=2048: check whether the
effect is actually in spike duration/frequency/ramp-rate rather than magnitude.
Reports PER-TRIAL (not pooled) so a monotonic budget trend can be checked for consistency
across trials rather than being an artifact of pooling."""
import csv
import os

DATE = os.environ.get("DATE", "2026-07-24")
NAME = os.environ.get("NAME", "pesim_gate")
BUDGETS = os.environ.get("BUDGETS", "16384 2048 512").split()
TRIALS = os.environ.get("TRIALS", "1 2 3").split()
LOGDIR = os.environ.get("LOGDIR", "logs")
THRESH_W = float(os.environ.get("THRESH_W", 420.0))  # "near-ceiling" threshold; override for
# setups whose power scale differs from the single-GPU 7B calibration this default was fit to
# (e.g. multi-GPU TP sums the whole group's power, so baseline/peak sit far above 420W there).


def load_power(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["wall_time"]), float(row["power_w"])))
    return rows


def stats_for(power):
    tot_samples = len(power)
    tot_above = 0
    max_ramp = 0.0
    ramp_sum = 0.0
    ramp_n = 0
    n_events = 0
    prev_above = False
    for i in range(tot_samples):
        t, p = power[i]
        above = p > THRESH_W
        tot_above += 1 if above else 0
        if above and not prev_above:
            n_events += 1
        prev_above = above
        if i > 0:
            dt = t - power[i - 1][0]
            dp = p - power[i - 1][1]
            if dt > 0:
                ramp = abs(dp / dt)
                ramp_sum += ramp
                ramp_n += 1
                max_ramp = max(max_ramp, ramp)
    frac_above = tot_above / tot_samples if tot_samples else float("nan")
    mean_ramp = ramp_sum / ramp_n if ramp_n else float("nan")
    return tot_samples, frac_above, n_events, mean_ramp, max_ramp


print(f"{'arm':<8}{'trial':<7}{'samples':<10}{'frac_above':<12}{'n_events':<10}{'mean_ramp':<12}{'max_ramp':<10}")
summary = {}
for b in BUDGETS:
    arm = f"b{b}"
    per_trial = []
    for tr in TRIALS:
        pw_path = f"{LOGDIR}/{DATE}-{NAME}-{arm}-t{tr}-power.csv"
        if not os.path.exists(pw_path):
            print(f"{arm:<8}{tr:<7} MISSING")
            continue
        power = load_power(pw_path)
        samples, frac_above, n_events, mean_ramp, max_ramp = stats_for(power)
        print(f"{arm:<8}{tr:<7}{samples:<10}{frac_above:<12.4f}{n_events:<10}{mean_ramp:<12.1f}{max_ramp:<10.1f}")
        per_trial.append((frac_above, n_events, mean_ramp, max_ramp))
    if per_trial:
        n = len(per_trial)
        summary[b] = tuple(sum(x[i] for x in per_trial) / n for i in range(4))

print()
print(f"{'arm':<8}{'frac_above (mean)':<20}{'n_events (mean)':<18}{'mean_ramp (mean)':<18}{'max_ramp (mean)':<16}")
for b in BUDGETS:
    if b in summary:
        fa, ne, mr, xr = summary[b]
        print(f"b{b:<7}{fa:<20.4f}{ne:<18.1f}{mr:<18.1f}{xr:<16.1f}")

print()
print("--- monotonicity check across trials (does mono > 2048 > 512 hold trial-by-trial for mean_ramp?) ---")
per_arm_per_trial = {}
for b in BUDGETS:
    arm = f"b{b}"
    per_arm_per_trial[b] = {}
    for tr in TRIALS:
        pw_path = f"{LOGDIR}/{DATE}-{NAME}-{arm}-t{tr}-power.csv"
        if os.path.exists(pw_path):
            power = load_power(pw_path)
            _, frac_above, n_events, mean_ramp, max_ramp = stats_for(power)
            per_arm_per_trial[b][tr] = (frac_above, n_events, mean_ramp, max_ramp)
for tr in TRIALS:
    row = []
    for b in BUDGETS:
        if tr in per_arm_per_trial.get(b, {}):
            row.append(f"b{b}:ramp={per_arm_per_trial[b][tr][2]:.1f},frac_above={per_arm_per_trial[b][tr][0]:.3f}")
    print(f"  trial {tr}: " + "  ".join(row))
