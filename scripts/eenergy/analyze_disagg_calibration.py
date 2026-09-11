"""Compute J/prefill-token and J/decode-token from the disaggregated calibration battery
(orchestrate/eenergy/run_disagg_prefill_decode_calibration.sh): GPU0's power trace is pure
prefill compute, GPU1's is pure decode compute, structurally (not inferred), for the entire
trial (idle-settle + burst + cooldown). Idle baseline is taken from the pre-burst settle
window; energy above that baseline, over the whole trace, divided by the burst's total
token counts, gives each rate directly -- no isolation-burst trick needed."""
import json
import sys

OUTNAME = "disagg_prefill_decode_calibration"


def load_power(path):
    times, powers = [], []
    with open(path) as f:
        next(f)  # header: wall_time,gpu_index,power_w,energy_mj,temp_c
        for line in f:
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3:
                continue
            times.append(float(parts[0]))
            powers.append(float(parts[2]))
    return times, powers


def trapz_energy_j(times, powers, t_lo, t_hi):
    """Integrate power (W) over [t_lo, t_hi] via trapezoidal rule -> energy in Joules."""
    energy = 0.0
    n = len(times)
    for i in range(1, n):
        t0, t1 = times[i - 1], times[i]
        if t1 < t_lo or t0 > t_hi:
            continue
        seg_lo = max(t0, t_lo)
        seg_hi = min(t1, t_hi)
        if seg_hi <= seg_lo:
            continue
        # linear interp of power at seg_lo/seg_hi
        if t1 > t0:
            frac_lo = (seg_lo - t0) / (t1 - t0)
            frac_hi = (seg_hi - t0) / (t1 - t0)
        else:
            frac_lo = frac_hi = 0.0
        p_lo = powers[i - 1] + frac_lo * (powers[i] - powers[i - 1])
        p_hi = powers[i - 1] + frac_hi * (powers[i] - powers[i - 1])
        energy += (p_lo + p_hi) / 2.0 * (seg_hi - seg_lo)
    return energy


def mean_power(times, powers, t_lo, t_hi):
    vals = [p for t, p in zip(times, powers) if t_lo <= t <= t_hi]
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    with open(f"logs/{OUTNAME}_timing.txt") as f:
        timing = {}
        for line in f:
            k, v = line.strip().split("=")
            timing[k] = float(v)
    burst_start = timing["burst_start_epoch"]
    burst_end = timing["burst_end_epoch"]

    t0_g0, p0_g0 = load_power(f"logs/{OUTNAME}_power_gpu0.csv")
    t0_g1, p0_g1 = load_power(f"logs/{OUTNAME}_power_gpu1.csv")

    idle_lo_g0, idle_hi_g0 = t0_g0[0], burst_start
    idle_lo_g1, idle_hi_g1 = t0_g1[0], burst_start
    idle_power_gpu0 = mean_power(t0_g0, p0_g0, idle_lo_g0, idle_hi_g0)
    idle_power_gpu1 = mean_power(t0_g1, p0_g1, idle_lo_g1, idle_hi_g1)

    trial_hi_g0 = t0_g0[-1]
    trial_hi_g1 = t0_g1[-1]

    raw_energy_gpu0_j = trapz_energy_j(t0_g0, p0_g0, burst_start, trial_hi_g0)
    raw_energy_gpu1_j = trapz_energy_j(t0_g1, p0_g1, burst_start, trial_hi_g1)
    duration_gpu0 = trial_hi_g0 - burst_start
    duration_gpu1 = trial_hi_g1 - burst_start
    idle_energy_gpu0_j = idle_power_gpu0 * duration_gpu0
    idle_energy_gpu1_j = idle_power_gpu1 * duration_gpu1
    above_idle_gpu0_j = raw_energy_gpu0_j - idle_energy_gpu0_j
    above_idle_gpu1_j = raw_energy_gpu1_j - idle_energy_gpu1_j

    prompt_tokens = 0
    decode_tokens = 0
    n_completed = 0
    n_failed = 0
    with open(f"logs/{OUTNAME}_records.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if not isinstance(rec.get("ttft"), (int, float)):
                n_failed += 1
                continue
            n_completed += 1
            prompt_tokens += rec.get("prompt_tokens_approx", 0)
            decode_tokens += rec.get("output_tokens", 0)

    j_per_prefill_token = above_idle_gpu0_j / prompt_tokens if prompt_tokens else float("nan")
    j_per_decode_token = above_idle_gpu1_j / decode_tokens if decode_tokens else float("nan")

    print(f"n_completed={n_completed} n_failed={n_failed}")
    print(f"prompt_tokens_total={prompt_tokens} decode_tokens_total={decode_tokens}")
    print(f"burst_duration_s={burst_end - burst_start:.1f}")
    print()
    print(f"GPU0 (prefill): idle_power={idle_power_gpu0:.1f}W  "
          f"above_idle_energy={above_idle_gpu0_j:.1f}J over {duration_gpu0:.1f}s")
    print(f"GPU1 (decode):  idle_power={idle_power_gpu1:.1f}W  "
          f"above_idle_energy={above_idle_gpu1_j:.1f}J over {duration_gpu1:.1f}s")
    print()
    print(f"j_per_prefill_token = {j_per_prefill_token:.4f}  (colocated calibration: 0.068)")
    print(f"j_per_decode_token  = {j_per_decode_token:.4f}  (colocated calibration: 2.40)")


if __name__ == "__main__":
    sys.exit(main())
