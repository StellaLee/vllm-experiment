"""Analyzes the isolation calibration run (orchestrate/eenergy/run_prefill_decode_calibration.sh,
3 trials within one continuous power trace, each trial drawing different conversations AND
measuring its own fresh idle baseline immediately before its prefill burst -- see the script's
header comment for why a single fixed pre-experiment idle baseline was replaced: the first
replicated run showed j_per_prefill_token was not robust (CV=76.8%, monotonic drift across
trials) because 15s settle windows never actually reached true idle, and residual power was
itself drifting down across the ~3min experiment).

Each trial's own fresh idle measurement is used for BOTH that trial's prefill and decode
subtraction -- removes the staleness/drift problem instead of just lengthening the old fixed
window. Reports per-trial numbers plus mean/std/CV% across trials, and a within-idle-window
stability check (first half vs second half of each trial's 45s idle window) as a diagnostic on
whether 45s is now actually long enough.
"""
import csv
import json
import sys
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "calibration"
OUTNAME = "prefill_decode_calibration"
TRIALS = (1, 2, 3)


def load_phases():
    phases = {}
    with open(f"{LOGDIR}/{PREFIX}_phases_{OUTNAME}.txt") as f:
        for line in f:
            name, ts = line.split()
            phases[name] = float(ts)
    return phases


def load_power_series():
    by_time = defaultdict(float)
    with open(f"{LOGDIR}/{PREFIX}_power_trace_{OUTNAME}.csv") as f:
        for row in csv.DictReader(f):
            by_time[float(row["wall_time"])] += float(row["power_w"])
    return sorted(by_time.items())


def window_energy_j(series, t0, t1):
    pts = [(t, p) for t, p in series if t0 <= t <= t1]
    if len(pts) < 2:
        return 0.0, 0.0
    energy = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        energy += 0.5 * (pa + pb) * (tb - ta)
    duration = pts[-1][0] - pts[0][0]
    return energy, duration


def load_tokens(records_path):
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


def mean(vals):
    return sum(vals) / len(vals)


def std(vals):
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    return (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5


def main():
    phases = load_phases()
    series = load_power_series()

    print("idle-window stability check (first half vs second half of each trial's 45s idle "
          "window -- large gaps mean 45s still isn't reaching true steady state):")
    for trial in TRIALS:
        t0, t1 = phases[f"trial{trial}_idle_start"], phases[f"trial{trial}_idle_end"]
        mid = (t0 + t1) / 2
        e1, d1 = window_energy_j(series, t0, mid)
        e2, d2 = window_energy_j(series, mid, t1)
        p1 = e1 / d1 if d1 else float("nan")
        p2 = e2 / d2 if d2 else float("nan")
        print(f"  trial{trial}: first_half={p1:.1f}W  second_half={p2:.1f}W  "
              f"delta={100*(p2-p1)/p1:.1f}%")

    prefill_rates, decode_rates, ratios = [], [], []
    for trial in TRIALS:
        idle_energy, idle_duration = window_energy_j(
            series, phases[f"trial{trial}_idle_start"], phases[f"trial{trial}_idle_end"])
        idle_power_w = idle_energy / idle_duration if idle_duration > 0 else 0.0

        prefill_energy, prefill_duration = window_energy_j(
            series, phases[f"trial{trial}_prefill_start"], phases[f"trial{trial}_prefill_end"])
        prefill_elevated_j = prefill_energy - idle_power_w * prefill_duration
        prefill_tokens, _ = load_tokens(f"{LOGDIR}/{PREFIX}_records_prefill_burst_t{trial}.jsonl")
        j_per_prefill_token = prefill_elevated_j / prefill_tokens if prefill_tokens else float("nan")

        decode_energy, decode_duration = window_energy_j(
            series, phases[f"trial{trial}_decode_start"], phases[f"trial{trial}_decode_end"])
        decode_idle_component = idle_power_w * decode_duration
        decode_prompt_tokens, decode_output_tokens = load_tokens(
            f"{LOGDIR}/{PREFIX}_records_decode_burst_t{trial}.jsonl")
        decode_prefill_component = j_per_prefill_token * decode_prompt_tokens
        decode_elevated_j = decode_energy - decode_idle_component - decode_prefill_component
        j_per_decode_token = (decode_elevated_j / decode_output_tokens
                               if decode_output_tokens else float("nan"))

        ratio = j_per_decode_token / j_per_prefill_token
        prefill_rates.append(j_per_prefill_token)
        decode_rates.append(j_per_decode_token)
        ratios.append(ratio)

        print(f"\ntrial {trial}: idle={idle_power_w:.1f}W  prefill_tokens={prefill_tokens} "
              f"j_per_prefill_token={j_per_prefill_token:.6f} J/token  "
              f"decode_tokens={decode_output_tokens} "
              f"j_per_decode_token={j_per_decode_token:.6f} J/token  ratio={ratio:.2f}x")

    print(f"\n{'metric':<20}{'mean':>12}{'std':>12}{'cv%':>10}")
    for label, vals in (("j_per_prefill_token", prefill_rates),
                         ("j_per_decode_token", decode_rates),
                         ("ratio", ratios)):
        m, s = mean(vals), std(vals)
        cv = 100.0 * s / m if m else float("nan")
        print(f"{label:<20}{m:>12.4f}{s:>12.4f}{cv:>9.1f}%")


if __name__ == "__main__":
    sys.exit(main())
