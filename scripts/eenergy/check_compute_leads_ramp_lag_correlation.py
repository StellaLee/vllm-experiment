"""Does incoming compute burst LEAD ramp_rate, or does ramp_rate merely lag it?

Motivation (see findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md, "does
power-awareness help" thread): drf_no_power (a genuinely power-blind 2-resource DRF)
dominates or matches coincidence_ceiling on several conditions, and only clearly wins on
Heavy/Closed-Loop's sustained, correlated-pressure regime. The proposed mechanistic
explanation: ramp_rate = dP/dt observed NOW reflects a compute step that landed roughly one
time-constant AGO (a lagging indicator), whereas share_compute at decision time is "how much
compute is about to land" (a leading indicator of the NEXT ramp event). If that's right,
incoming compute at time t should cross-correlate most strongly with ramp_rate at some POSITIVE
lag (ramp follows compute), not lag 0 or negative -- and DRF's compute-balancing already
exploits the leading signal directly, which a rule reading raw ramp_rate cannot.

This is a read-only check against already-collected Heavy/Closed-Loop-long hardware logs
(coincidence_ceiling, t1-t6) -- no new experiments needed.

Method, per (trial, gpu) pair:
  1. power_trace -> ramp_rate(t) = diff(power_w) / diff(wall_time), resampled onto a uniform
     1s grid via linear interpolation of the cumulative power_w signal then differencing (the
     raw power_trace is already ~1Hz per gpu but not perfectly uniformly spaced across gpus).
  2. assignment log -> incoming compute rate(t) = sum(new_tokens) dispatched to that gpu in
     each 1s bin (a direct proxy for compute burst arriving at that replica).
  3. Cross-correlate the two z-scored signals at integer-second lags from -20s to +20s
     (positive lag = compute leads ramp by that many seconds, i.e. corr(compute[t],
     ramp[t+lag])).
  4. Pool the per-(trial,gpu) correlation curves (mean across the 6 trials x 6 gpus = 36
     series) and report the lag of peak pooled correlation plus its value, alongside lag=0
     and the most-negative lag tested, to see whether the peak is displaced toward positive
     lag (compute-leads) as predicted.
"""
import sys
import numpy as np
import pandas as pd

LOG_DIR = "logs"
TRIALS = range(1, 7)
POLICY = "coincidence_ceiling"
PREFIX = "closedloopheavylongpergpu"
MAX_LAG_S = 20


def load_trial(trial: int):
    power = pd.read_csv(f"{LOG_DIR}/{PREFIX}_power_trace_{POLICY}_t{trial}.csv")
    assign = pd.read_csv(f"{LOG_DIR}/{PREFIX}_assignment_{POLICY}_t{trial}.csv")
    return power, assign


def per_gpu_series(power: pd.DataFrame, assign: pd.DataFrame, gpu: int):
    p = power[power.gpu_index == gpu].sort_values("wall_time")
    a = assign[assign.gpu_index == gpu].sort_values("wall_time")
    if len(p) < 10 or len(a) < 5:
        return None
    t0 = p.wall_time.min()
    t1 = p.wall_time.max()
    grid = np.arange(np.floor(t0), np.ceil(t1) + 1.0, 1.0)
    # cumulative power interpolated onto uniform grid, then diff -> ramp on a clean 1Hz grid
    power_interp = np.interp(grid, p.wall_time.values, p.power_w.values)
    ramp = np.diff(power_interp, prepend=power_interp[0])  # W/s at 1s spacing == W/s directly

    bins = np.floor(a.wall_time.values - t0).astype(int)
    compute_rate = np.zeros(len(grid))
    for b, tok in zip(bins, a.new_tokens.values):
        if 0 <= b < len(compute_rate):
            compute_rate[b] += tok

    return grid, ramp, compute_rate


def zscore(x):
    s = x.std()
    if s < 1e-9:
        return None
    return (x - x.mean()) / s


def lagged_corr(compute_rate, ramp, max_lag):
    """corr(compute[t], ramp[t+lag]) for lag in [-max_lag, max_lag]."""
    n = len(compute_rate)
    c = zscore(compute_rate)
    r = zscore(ramp)
    if c is None or r is None:
        return None
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a_seg = c[: n - lag] if lag > 0 else c
            b_seg = r[lag:]
        else:
            a_seg = c[-lag:]
            b_seg = r[: n + lag]
        if len(a_seg) < 20:
            continue
        out[lag] = float(np.corrcoef(a_seg, b_seg)[0, 1])
    return out


def main():
    all_curves = []
    n_series = 0
    for trial in TRIALS:
        power, assign = load_trial(trial)
        gpus = sorted(power.gpu_index.unique())
        for gpu in gpus:
            series = per_gpu_series(power, assign, gpu)
            if series is None:
                continue
            grid, ramp, compute_rate = series
            curve = lagged_corr(compute_rate, ramp, MAX_LAG_S)
            if curve is None:
                continue
            all_curves.append(curve)
            n_series += 1

    print(f"pooled series (trial x gpu pairs): {n_series}")
    lags = list(range(-MAX_LAG_S, MAX_LAG_S + 1))
    pooled = {}
    for lag in lags:
        vals = [c[lag] for c in all_curves if lag in c]
        if vals:
            pooled[lag] = (float(np.mean(vals)), float(np.std(vals)), len(vals))

    print(f"{'lag(s)':>7} {'mean_r':>8} {'std_r':>7} {'n':>4}")
    for lag in lags:
        if lag in pooled:
            m, s, n = pooled[lag]
            marker = "  <-- lag 0" if lag == 0 else ""
            print(f"{lag:>7} {m:>8.4f} {s:>7.4f} {n:>4}{marker}")

    best_lag = max(pooled, key=lambda l: pooled[l][0])
    print()
    print(f"peak pooled correlation at lag={best_lag}s: r={pooled[best_lag][0]:.4f} "
          f"(n={pooled[best_lag][2]})")
    print(f"lag=0: r={pooled[0][0]:.4f}")
    if best_lag > 0:
        print("Peak is at POSITIVE lag: compute leads ramp (consistent with the "
              "leading-indicator hypothesis).")
    elif best_lag < 0:
        print("Peak is at NEGATIVE lag: ramp leads compute (inconsistent with the "
              "hypothesis as stated).")
    else:
        print("Peak is at lag 0: compute and ramp are contemporaneous, no clear lead/lag.")


if __name__ == "__main__":
    sys.exit(main())
