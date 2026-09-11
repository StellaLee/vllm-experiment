"""Quick spike: does a rolling-window energy-budget ADMISSION gate look practically tolerable
on real traffic, or does it impose so much deferral that it's not worth building for real?

Uses round_robin's real request-arrival trace (assignment log: real wall_time + real token
counts per request) against that SAME trial's real, historical fleet-aggregate power trace as
ground truth for "how loaded is the fleet right now." For each request dispatch, checks
whether the trailing W-second average fleet power at that instant already exceeds cap C; if
so, estimates how long it would need to wait (scanning forward in the real trace) until the
rolling average drops back under C.

This is an approximation, not a full re-simulation: it doesn't account for how deferring a
request would itself change the future power trace (the real trace already includes that
request's actual contribution). It answers a narrower, cheaper question first: given the
REAL demand pattern this fleet already produced, how often and how long would admission need
to hold requests back to keep a rolling average under various caps? If that's rare/short,
full re-simulation + a live implementation is worth the effort. If it's frequent/long, this
particular cap/window combination isn't practically viable and should be revised before
investing further.

Routing (which replica an admitted request goes to, i.e. the DRF compute/load balance) is
orthogonal to this admission-only question and intentionally not simulated here.
"""
import csv
import sys
from bisect import bisect_left

sys.path.insert(0, "/root/pli/vllm-experiment/scripts/eenergy")
import compare_closedloopheavy_duration as c  # noqa: E402

LOGDIR = "/root/pli/vllm-experiment/logs"


def load_dispatch_times(assign_path):
    times = []
    with open(assign_path) as f:
        r = csv.DictReader(f)
        for row in r:
            times.append(float(row["wall_time"]))
    return sorted(times)


def rolling_avg_at(series_t, series_p, t, window_s):
    """Trailing window_s-second average fleet power ending at time t, from the real series."""
    lo = bisect_left(series_t, t - window_s)
    hi = bisect_left(series_t, t) + 1
    hi = min(hi, len(series_t))
    if hi - lo < 2:
        return series_p[hi - 1] if hi > 0 else 0.0
    vals = series_p[lo:hi]
    return sum(vals) / len(vals)


def simulate(prefix, arm, trial, cap_w, window_s, label):
    assign_path = f"{LOGDIR}/{prefix}_assignment_{arm}_t{trial}.csv"
    pow_path = f"{LOGDIR}/{prefix}_power_trace_{arm}_t{trial}.csv"
    dispatch_times = load_dispatch_times(assign_path)
    series = c.load_power(pow_path)
    series_t = [t for t, _ in series]
    series_p = [p for _, p in series]
    trace_end = series_t[-1]

    n_total = len(dispatch_times)
    n_deferred = 0
    wait_times = []

    for t in dispatch_times:
        avg_now = rolling_avg_at(series_t, series_p, t, window_s)
        if avg_now <= cap_w:
            continue
        n_deferred += 1
        # scan forward in the real trace for the next point where the rolling avg drops <= cap
        idx = bisect_left(series_t, t)
        waited = None
        for j in range(idx, len(series_t)):
            if rolling_avg_at(series_t, series_p, series_t[j], window_s) <= cap_w:
                waited = series_t[j] - t
                break
        if waited is None:
            waited = trace_end - t  # never recovers before trace ends -- upper bound
        wait_times.append(waited)

    print(f"\n=== {label}: cap={cap_w}W window={window_s}s ===")
    print(f"requests: {n_total}   would-need-deferral: {n_deferred} "
          f"({100.0 * n_deferred / n_total:.1f}%)")
    if wait_times:
        wait_times.sort()
        n = len(wait_times)
        mean_w = sum(wait_times) / n
        p50 = wait_times[n // 2]
        p95 = wait_times[min(n - 1, int(n * 0.95))]
        mx = wait_times[-1]
        print(f"deferral wait (s): mean={mean_w:.2f}  p50={p50:.2f}  p95={p95:.2f}  max={mx:.2f}")
    else:
        print("no deferrals needed")


def main():
    for cap in (2200, 2400):
        simulate("closedloopheavylongpergpu", "round_robin", 1, cap, 15,
                  "Heavy/CL long (sustained)")
    for cap in (2400, 2200):
        simulate("openloopwhalelongoutmatchedpergpu", "round_robin", 1, cap, 30,
                  "Heavy/Matched (bursty)")


if __name__ == "__main__":
    sys.exit(main())
