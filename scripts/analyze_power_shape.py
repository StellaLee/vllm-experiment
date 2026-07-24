#!/usr/bin/env python3
"""Diagnose WHY peak power didn't differ between mono and chunk=2048: check whether the
effect is actually in spike duration/frequency/ramp-rate rather than magnitude."""
import csv
import json
import os
import sys

DATE = "2026-07-24"
BUDGETS = ["16384", "2048", "512"]
TRIALS = ["1", "2", "3"]
THRESH_W = 420.0  # "near-ceiling" threshold


def load_power(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["wall_time"]), float(row["power_w"])))
    return rows


for b in BUDGETS:
    arm = f"b{b}"
    tot_samples = 0
    tot_above = 0
    max_ramp = 0.0
    ramp_sum = 0.0
    ramp_n = 0
    n_events = 0  # count of rising-edge crossings above threshold (spike frequency)
    for tr in TRIALS:
        pw_path = f"logs/{DATE}-pesimgate-{arm}-t{tr}-power.csv"
        if not os.path.exists(pw_path):
            continue
        power = load_power(pw_path)
        tot_samples += len(power)
        prev_above = False
        for i in range(len(power)):
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
    print(f"{arm}: samples={tot_samples} frac_time_above_{THRESH_W}W={frac_above:.4f} "
          f"n_spike_events={n_events} mean_ramp_W_per_s={mean_ramp:.1f} max_ramp_W_per_s={max_ramp:.1f}")
