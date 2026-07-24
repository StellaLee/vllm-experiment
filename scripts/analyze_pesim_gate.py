#!/usr/bin/env python3
"""Analyze the PES-IM gate experiment: whale-prefill power spike (C3) + energy
conservation (C1) across mono(16384)/chunk(2048)/chunk(512) on 7B/single-GPU.

For each (budget, trial):
  - loads the request-record jsonl (has ts=completion wall-time, latency, ttft, pad_chars)
  - loads the aligned power-trace csv (wall_time, power_w, energy_mj, temp_c)
  - identifies whale requests (pad_chars >= WHALE_THRESH) and their prefill window
    [start, start+ttft] where start = ts - latency
  - baseline power = median of samples NOT inside any whale prefill window
  - peak power = max power sample inside any whale prefill window
  - energy-per-output-token = (energy_mj[-1] - energy_mj[0]) / sum(output_tokens), a first-order
    proxy for C1 (energy roughly invariant across scheduling policy) -- NOTE: denominator is
    output tokens only (prompt tokens not exactly counted, only word-count approx available),
    so this is a relative cross-arm comparison, not an absolute energy/token figure.
"""
import csv
import json
import os

WHALE_THRESH = 40000  # pad_chars cutoff, consistent with the 14B longprompt convention

DATE = os.environ.get("DATE")
BUDGETS = os.environ.get("BUDGETS", "16384 2048 512").split()
TRIALS = os.environ.get("TRIALS", "1").split()


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
            rows.append({
                "t": float(row["wall_time"]),
                "power_w": float(row["power_w"]),
                "energy_mj": int(row["energy_mj"]),
                "temp_c": float(row["temp_c"]),
            })
    return rows


def whale_windows(recs):
    windows = []
    for r in recs:
        if r.get("pad_chars", 0) >= WHALE_THRESH and r.get("ts") is not None and r.get("latency") is not None:
            end = r["ts"]
            start = end - r["latency"]
            ttft = r.get("ttft") or 0
            windows.append((start, start + ttft))
    return windows


def in_any_window(t, windows):
    for s, e in windows:
        if s <= t <= e:
            return True
    return False


def main():
    print(f"{'arm':<8}{'trial':<7}{'n':<6}{'nwhale':<8}{'baseline_W':<12}{'peak_W':<10}{'spike_x':<10}{'kJ/1k_outtok':<14}")
    summary = {}
    for b in BUDGETS:
        arm = f"b{b}"
        arm_baselines, arm_peaks, arm_ept = [], [], []
        for tr in TRIALS:
            rec_path = f"logs/{DATE}-pesimgate-{arm}-t{tr}.jsonl"
            pw_path = f"logs/{DATE}-pesimgate-{arm}-t{tr}-power.csv"
            if not (os.path.exists(rec_path) and os.path.exists(pw_path)):
                print(f"{arm:<8}{tr:<7} MISSING ({rec_path} or {pw_path})")
                continue
            recs = load_records(rec_path)
            power = load_power(pw_path)
            if not recs or not power:
                print(f"{arm:<8}{tr:<7} EMPTY")
                continue
            windows = whale_windows(recs)
            nwhale = len(windows)

            baseline_samples = [p["power_w"] for p in power if not in_any_window(p["t"], windows)]
            whale_samples = [p["power_w"] for p in power if in_any_window(p["t"], windows)]
            baseline_w = sorted(baseline_samples)[len(baseline_samples) // 2] if baseline_samples else float("nan")
            peak_w = max(whale_samples) if whale_samples else float("nan")
            spike_x = peak_w / baseline_w if baseline_w else float("nan")

            total_output_tok = sum(r.get("output_tokens", 0) or 0 for r in recs)
            energy_mj_delta = power[-1]["energy_mj"] - power[0]["energy_mj"]
            kj_per_1k_outtok = (energy_mj_delta / 1000.0) / (total_output_tok / 1000.0) if total_output_tok else float("nan")

            print(f"{arm:<8}{tr:<7}{len(recs):<6}{nwhale:<8}{baseline_w:<12.1f}{peak_w:<10.1f}{spike_x:<10.2f}{kj_per_1k_outtok:<14.2f}")
            arm_baselines.append(baseline_w)
            arm_peaks.append(peak_w)
            arm_ept.append(kj_per_1k_outtok)
        if arm_peaks:
            summary[b] = {
                "baseline_w": sum(arm_baselines) / len(arm_baselines),
                "peak_w": max(arm_peaks),
                "kj_per_1k_outtok": sum(arm_ept) / len(arm_ept),
            }

    print()
    print("--- summary (mean over trials, peak = max over trials) ---")
    print(f"{'budget':<10}{'baseline_W':<12}{'peak_W':<10}{'kJ/1k_outtok':<14}")
    for b in BUDGETS:
        if b in summary:
            s = summary[b]
            print(f"{b:<10}{s['baseline_w']:<12.1f}{s['peak_w']:<10.1f}{s['kj_per_1k_outtok']:<14.2f}")

    if len(BUDGETS) >= 2 and all(b in summary for b in BUDGETS):
        mono = BUDGETS[0]
        print()
        print(f"--- go/no-go: does chunking flatten the peak vs mono({mono}) while energy/token stays flat? ---")
        for b in BUDGETS[1:]:
            dpeak = (summary[b]["peak_w"] - summary[mono]["peak_w"]) / summary[mono]["peak_w"] * 100
            depk = (summary[b]["kj_per_1k_outtok"] - summary[mono]["kj_per_1k_outtok"]) / summary[mono]["kj_per_1k_outtok"] * 100
            print(f"  {b} vs {mono}: peak power {dpeak:+.1f}%, energy/token {depk:+.1f}%")


if __name__ == "__main__":
    main()
