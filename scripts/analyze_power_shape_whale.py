#!/usr/bin/env python3
"""Disaggregate the ramp-rate/duty-cycle effect (found in analyze_power_shape.py) into
whale-transition windows vs. the rest of the trace, to confirm the effect is concentrated
where the mechanism predicts (whale prefill onset/offset) rather than being a diffuse
artifact spread across the whole run.

A "whale window" is [ts - latency, ts - latency + ttft] (the whale's prefill duration) per
whale request, widened by WINDOW_PAD_S on each side to catch the transition itself (ramp up
just before the window starts, ramp down just after it ends).
"""
import csv
import json
import os

DATE = os.environ.get("DATE", "2026-07-24")
NAME = os.environ.get("NAME", "pesim_gate")
BUDGETS = os.environ.get("BUDGETS", "16384 2048 512").split()
TRIALS = os.environ.get("TRIALS", "1 2 3").split()
THRESH_W = 420.0
WHALE_THRESH_CHARS = int(os.environ.get("WHALE_THRESH_CHARS", 40000))
WINDOW_PAD_S = 1.0  # widen whale windows by this much on each side to catch the transition


def load_records(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def load_power(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((float(row["wall_time"]), float(row["power_w"])))
    return rows


def whale_windows(recs):
    windows = []
    for r in recs:
        if r.get("pad_chars", 0) >= WHALE_THRESH_CHARS and r.get("ts") is not None and r.get("latency") is not None:
            end = r["ts"]
            start = end - r["latency"]
            ttft = r.get("ttft") or 0
            windows.append((start - WINDOW_PAD_S, start + ttft + WINDOW_PAD_S))
    return windows


def in_any_window(t, windows):
    for s, e in windows:
        if s <= t <= e:
            return True
    return False


def stats_for(samples):
    # samples: list of (t, p) already filtered to the region of interest
    n = len(samples)
    if n == 0:
        return 0, float("nan"), 0, float("nan")
    tot_above = 0
    max_ramp = 0.0
    ramp_sum = 0.0
    ramp_n = 0
    n_events = 0
    prev_above = False
    for i in range(n):
        t, p = samples[i]
        above = p > THRESH_W
        tot_above += 1 if above else 0
        if above and not prev_above:
            n_events += 1
        prev_above = above
        if i > 0:
            dt = t - samples[i - 1][0]
            dp = p - samples[i - 1][1]
            if 0 < dt <= 0.5:  # skip gaps (e.g. across the in/out-of-window split)
                ramp_sum += abs(dp / dt)
                ramp_n += 1
                max_ramp = max(max_ramp, abs(dp / dt))
    frac_above = tot_above / n
    mean_ramp = ramp_sum / ramp_n if ramp_n else float("nan")
    return n, frac_above, n_events, mean_ramp


print(f"{'arm':<8}{'trial':<7}{'region':<10}{'n':<8}{'frac_above':<12}{'n_events':<10}{'mean_ramp':<10}")
agg = {}  # (b, region) -> list of (frac_above, mean_ramp) across trials
for b in BUDGETS:
    arm = f"b{b}"
    for tr in TRIALS:
        rec_path = f"logs/{DATE}-{NAME}-{arm}-t{tr}.jsonl"
        pw_path = f"logs/{DATE}-{NAME}-{arm}-t{tr}-power.csv"
        if not (os.path.exists(rec_path) and os.path.exists(pw_path)):
            continue
        recs = load_records(rec_path)
        power = load_power(pw_path)
        windows = whale_windows(recs)
        in_whale = [p for p in power if in_any_window(p[0], windows)]
        out_whale = [p for p in power if not in_any_window(p[0], windows)]
        for region, samples in [("whale", in_whale), ("non-whale", out_whale)]:
            n, frac_above, n_events, mean_ramp = stats_for(samples)
            print(f"{arm:<8}{tr:<7}{region:<10}{n:<8}{frac_above:<12.4f}{n_events:<10}{mean_ramp:<10.1f}")
            agg.setdefault((b, region), []).append((frac_above, mean_ramp))

print()
print(f"{'arm':<8}{'region':<10}{'frac_above (mean)':<20}{'mean_ramp (mean)':<18}")
for b in BUDGETS:
    for region in ("whale", "non-whale"):
        vals = agg.get((b, region), [])
        if vals:
            fa = sum(v[0] for v in vals) / len(vals)
            mr = sum(v[1] for v in vals) / len(vals)
            print(f"b{b:<7}{region:<10}{fa:<20.4f}{mr:<18.1f}")
