"""Analyzes the isolation calibration run (orchestrate/eenergy/run_prefill_decode_calibration.sh):
slices the one continuous power trace by phase boundary, computes idle baseline power, then
computes j_per_prefill_token from the near-zero-decode prefill burst and j_per_decode_token
from the decode burst (subtracting BOTH idle power and the prefill contribution of that
phase's own, much shorter, natural-length prompts, using the just-calibrated prefill rate).
"""
import csv
import json
import sys
from collections import defaultdict

LOGDIR = "/root/pli/vllm-experiment/logs"
PREFIX = "calibration"
OUTNAME = "prefill_decode_calibration"


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
    """Trapezoidal-ish integral of fleet power over [t0, t1] using consecutive samples."""
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


def main():
    phases = load_phases()
    series = load_power_series()

    idle_energy, idle_duration = window_energy_j(series, phases["idle_start"], phases["idle_end"])
    idle_power_w = idle_energy / idle_duration if idle_duration > 0 else 0.0
    print(f"idle baseline: {idle_power_w:.1f} W (over {idle_duration:.1f}s)")

    prefill_energy, prefill_duration = window_energy_j(
        series, phases["prefill_start"], phases["prefill_end"])
    prefill_idle_component = idle_power_w * prefill_duration
    prefill_elevated_j = prefill_energy - prefill_idle_component
    prefill_tokens, prefill_phase_output_tokens = load_tokens(
        f"{LOGDIR}/{PREFIX}_records_prefill_burst.jsonl")
    j_per_prefill_token = prefill_elevated_j / prefill_tokens if prefill_tokens else float("nan")
    print(f"\nprefill phase: {prefill_duration:.1f}s, energy={prefill_energy:.1f}J, "
          f"idle_component={prefill_idle_component:.1f}J, elevated={prefill_elevated_j:.1f}J")
    print(f"prefill tokens={prefill_tokens}, output tokens (should be ~= num requests, "
          f"max_tokens=1)={prefill_phase_output_tokens}")
    print(f"j_per_prefill_token = {j_per_prefill_token:.6f} J/token")

    decode_energy, decode_duration = window_energy_j(
        series, phases["decode_start"], phases["decode_end"])
    decode_idle_component = idle_power_w * decode_duration
    decode_prompt_tokens, decode_output_tokens = load_tokens(
        f"{LOGDIR}/{PREFIX}_records_decode_burst.jsonl")
    decode_prefill_component = j_per_prefill_token * decode_prompt_tokens
    decode_elevated_j = decode_energy - decode_idle_component - decode_prefill_component
    j_per_decode_token = decode_elevated_j / decode_output_tokens if decode_output_tokens else float("nan")
    print(f"\ndecode phase: {decode_duration:.1f}s, energy={decode_energy:.1f}J, "
          f"idle_component={decode_idle_component:.1f}J, "
          f"prefill_component={decode_prefill_component:.1f}J (from {decode_prompt_tokens} "
          f"natural-prompt tokens), elevated={decode_elevated_j:.1f}J")
    print(f"decode output tokens={decode_output_tokens}")
    print(f"j_per_decode_token = {j_per_decode_token:.6f} J/token")

    print(f"\nratio j_per_decode_token / j_per_prefill_token = "
          f"{j_per_decode_token / j_per_prefill_token:.2f}x")


if __name__ == "__main__":
    sys.exit(main())
