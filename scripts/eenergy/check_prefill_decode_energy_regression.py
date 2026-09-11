"""Fits total_energy_j ~= j_per_prefill_token * total_prompt_tokens
                          + j_per_decode_token * total_output_tokens
via least squares across many already-collected trials (varying prefill:decode ratios across
conditions), to check whether a genuinely separated prefill/decode energy model is
well-determined by existing data -- a prerequisite before building a live estimator around it.
No new hardware trials; reuses logs already on disk.
"""
import csv
import glob
import json
import re

import numpy as np


def energy_j_and_duration_s(power_trace_path):
    first, last = {}, {}
    t0, t1 = None, None
    with open(power_trace_path) as f:
        for row in csv.DictReader(f):
            gpu = row["gpu_index"]
            e = float(row["energy_mj"])
            t = float(row["wall_time"])
            if gpu not in first:
                first[gpu] = e
            last[gpu] = e
            t0 = t if t0 is None else min(t0, t)
            t1 = t if t1 is None else max(t1, t)
    return sum((last[g] - first[g]) / 1000.0 for g in first), (t1 - t0)


def tokens(records_path):
    prompt_tokens = 0
    output_tokens = 0
    with open(records_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            prompt_tokens += r.get("prompt_tokens_approx", 0) or 0
            output_tokens += r.get("output_tokens", 0) or 0
    return prompt_tokens, output_tokens


def main():
    rows = []
    for pow_path in glob.glob("logs/*pergpu_power_trace_*.csv"):
        m = re.match(r"logs/(.+)_power_trace_(.+)_t(\d+)\.csv", pow_path)
        if not m:
            continue
        prefix, arm, trial = m.groups()
        rec_path = f"logs/{prefix}_records_{arm}_t{trial}.jsonl"
        try:
            e, duration_s = energy_j_and_duration_s(pow_path)
            pt, ot = tokens(rec_path)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        if pt == 0 and ot == 0:
            continue
        rows.append((prefix, arm, trial, pt, ot, duration_s, e))

    print(f"usable trials: {len(rows)}")
    if len(rows) < 10:
        print("too few trials for a trustworthy fit -- stopping here")
        return

    for label, use_duration in (("2-var (tokens only)", False), ("3-var (+ idle*duration)", True)):
        if use_duration:
            X = np.array([[pt, ot, dur] for _, _, _, pt, ot, dur, _ in rows], dtype=float)
        else:
            X = np.array([[pt, ot] for _, _, _, pt, ot, _, _ in rows], dtype=float)
        y = np.array([e for *_, e in rows], dtype=float)

        # Least squares, no separate intercept (duration itself, when included, plays that
        # role -- a fleet with zero tokens and zero duration should draw zero energy).
        coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ coeffs
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot
        abs_pct_err = np.abs((pred - y) / y) * 100.0

        print(f"\n=== {label} ===")
        if use_duration:
            print(f"j_per_prefill_token = {coeffs[0]:.6f} J/token")
            print(f"j_per_decode_token  = {coeffs[1]:.6f} J/token")
            print(f"idle_fleet_power_w  = {coeffs[2]:.2f} W")
        else:
            print(f"j_per_prefill_token = {coeffs[0]:.6f} J/token")
            print(f"j_per_decode_token  = {coeffs[1]:.6f} J/token")
        print(f"R^2 = {r2:.4f}  (rank={rank})  "
              f"median_abs_err={np.median(abs_pct_err):.1f}%  "
              f"p90_abs_err={np.percentile(abs_pct_err, 90):.1f}%")

    corr = np.corrcoef([pt for _, _, _, pt, _, _, _ in rows],
                        [ot for _, _, _, _, ot, _, _ in rows])[0, 1]
    print(f"\ncorr(total_prompt_tokens, total_output_tokens) across trials = {corr:.4f}")


if __name__ == "__main__":
    main()
