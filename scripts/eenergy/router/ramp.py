"""Pure ramp-rate (W/s) arithmetic, kept separate from the actual NVML I/O (power_nvml.py)
so the math is unit-testable without real hardware or pynvml installed."""


def compute_ramp_rate(prev_power_w: float, prev_ts: float,
                       cur_power_w: float, cur_ts: float) -> float:
    """Instantaneous ramp rate between two power samples, W/s. Returns 0.0 for a
    zero/negative interval (out-of-order or duplicate timestamps) rather than dividing by
    zero or reporting a nonsensical sign flip."""
    dt = cur_ts - prev_ts
    if dt <= 0:
        return 0.0
    return (cur_power_w - prev_power_w) / dt


def update_ramp_state(state, power_w: float, ts: float, smoothing_alpha: float = 0.3) -> None:
    """Mutates a ReplicaState in place: computes ramp rate from the previous sample (0.0 if
    this is the first sample ever, i.e. last_power_ts is still its default 0.0), then
    stores the new sample as the previous one for next time.

    Also maintains smoothed_ramp_rate_w_per_s, an EMA of the same raw two-point estimate
    (smoothed = smoothing_alpha * raw + (1 - smoothing_alpha) * previous_smoothed), for
    scoring functions that prefer a less noise-sensitive live signal than the raw
    finite-difference derivative -- ramp_rate_w_per_s is a single-sample-jitter-sensitive
    estimate (subject to whatever timing noise the two consecutive power samples happened to
    have), while the smoothed field damps that at the cost of some lag in tracking a genuine
    fast transition. Computing both unconditionally (rather than only on request) keeps this
    function's behavior for ramp_rate_w_per_s and every existing consumer of it completely
    unchanged -- this only ADDS a field, never alters the existing one."""
    if state.last_power_ts > 0:
        raw = compute_ramp_rate(state.last_power_w, state.last_power_ts, power_w, ts)
        state.ramp_rate_w_per_s = raw
        state.smoothed_ramp_rate_w_per_s = (
            smoothing_alpha * raw + (1.0 - smoothing_alpha) * state.smoothed_ramp_rate_w_per_s)
    state.last_power_w = power_w
    state.last_power_ts = ts
