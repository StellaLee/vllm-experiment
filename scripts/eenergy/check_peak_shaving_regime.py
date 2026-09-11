"""Characterizes whether peak-power violations in this fleet's existing traces are burst-driven
(routing/scheduling alone could plausibly fix them by spreading correlated spikes out in time)
or sustained-overload-driven (no routing scheme fixes this without admission control/deferral).

Method: load a representative fleet-aggregate power trace (round_robin -- the "no power
management at all" baseline, so this characterizes the underlying WORKLOAD's natural demand
shape, not any existing routing policy's effect on it), then for several candidate peak caps,
compare the violation rate at raw instantaneous sampling vs. at several rolling-window averages
(5s/15s/30s/60s -- proxies for a utility demand-charge averaging window). If windowing sharply
cuts the violation rate, that's a burst-driven regime (favorable for a no-storage, routing-only
peak-shaving strategy). If violations persist even at long windows, that's sustained overload
(routing alone cannot satisfy that cap; needs deferral/admission control).
"""
import sys
from bisect import bisect_left, bisect_right

sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import compare_closedloopheavy_duration as c  # noqa: E402

LOGDIR = "/root/pli/vllm-experiment/logs"


def rolling_window_avg(series, window_s):
    """series: sorted list of (t, P) fleet-aggregate samples. Returns the average power over
    every window_s-second window, evaluated once per raw sample (trailing window ending at
    that sample's time) -- a simple, conservative proxy for a utility's rolling demand-charge
    window."""
    times = [t for t, _ in series]
    n = len(series)
    out = []
    # prefix sums over (t, P) pairs using trapezoidal-ish simple average of samples in window
    for i in range(n):
        t_end = times[i]
        t_start = t_end - window_s
        lo = bisect_left(times, t_start)
        hi = i + 1
        if hi - lo < 2:
            continue
        window_vals = [p for _, p in series[lo:hi]]
        out.append(sum(window_vals) / len(window_vals))
    return out


def analyze(prefix, arm, trial, label, caps):
    pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    series = c.load_power(pow_path)
    peak = max(p for _, p in series)
    mean = sum(p for _, p in series) / len(series)
    duration = series[-1][0] - series[0][0]
    n_samples = len(series)
    print(f"\n=== {label} ({prefix}, arm={arm}, t{trial}) ===")
    print(f"duration={duration:.1f}s  n_samples={n_samples}  "
          f"mean_spacing={duration / max(1, n_samples - 1):.2f}s")
    print(f"instantaneous: mean={mean:.1f}W  peak={peak:.1f}W  peak/mean={peak / mean:.2f}x")

    windows = {}
    for w in (5, 15, 30, 60):
        windows[w] = rolling_window_avg(series, w)

    print(f"\n{'cap(W)':>8}{'raw_viol%':>11}" + "".join(f"{'w='+str(w)+'s_viol%':>12}" for w in windows))
    for cap in caps:
        raw_viol = 100.0 * sum(1 for _, p in series if p > cap) / len(series)
        row = f"{cap:>8}{raw_viol:>10.1f}%"
        for w in windows:
            wv = windows[w]
            viol = 100.0 * sum(1 for p in wv if p > cap) / len(wv) if wv else float("nan")
            row += f"{viol:>11.1f}%"
        print(row)


def main():
    # Fleet ceiling context: 6 GPUs, per-GPU calibrated ramp ceiling implies ~450W-ish hardware
    # limits each -> theoretical max ~2700W. Caps chosen relative to that and to the peaks
    # actually observed (~2400-2650W across existing arms/conditions).
    caps = [2000, 2200, 2400, 2500, 2600]

    analyze("closedloopheavylongpergpu", "round_robin", 1, "Heavy/CL long (sustained, engineered pressure)", caps)
    analyze("openloopwhalelongoutmatchedpergpu", "round_robin", 1, "Heavy/Matched (bursty, open-loop)", caps)
    analyze("cachehitpergpu", "coincidence_ceiling", 1, "Light/Cachehit (light, no whale)", caps)


if __name__ == "__main__":
    sys.exit(main())
