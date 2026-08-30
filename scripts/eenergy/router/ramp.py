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


def update_ramp_state(state, power_w: float, ts: float) -> None:
    """Mutates a ReplicaState in place: computes ramp rate from the previous sample (0.0 if
    this is the first sample ever, i.e. last_power_ts is still its default 0.0), then
    stores the new sample as the previous one for next time."""
    if state.last_power_ts > 0:
        state.ramp_rate_w_per_s = compute_ramp_rate(
            state.last_power_w, state.last_power_ts, power_w, ts)
    state.last_power_w = power_w
    state.last_power_ts = ts
