"""Rolling-window fleet-power budget for the peak-shaving admission gate (see
docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md). Pure, hardware-free --
power_poll_loop (proxy_server.py) feeds real samples in; the admission gate in
handle_completions reads it before routing."""
from collections import deque


class PowerBudget:
    """Tracks a rolling window of fleet-aggregate power samples and answers whether admitting
    a request with a given marginal energy cost would push the trailing window's average
    power over a cap. Fails OPEN (would_exceed returns False) when the window has fewer than
    2 samples -- at startup, before power_poll_loop has produced enough history, blocking all
    traffic would be worse than the brief risk of an unenforced cap during warmup."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._samples = deque()  # (t, fleet_power_w), oldest first

    def record(self, t: float, fleet_power_w: float) -> None:
        self._samples.append((t, fleet_power_w))
        cutoff = t - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def would_exceed(self, cap_w: float, marginal_j: float) -> bool:
        if len(self._samples) < 2:
            return False
        mean_power_w = sum(p for _, p in self._samples) / len(self._samples)
        projected_avg_w = mean_power_w + marginal_j / self.window_s
        return projected_avg_w > cap_w
