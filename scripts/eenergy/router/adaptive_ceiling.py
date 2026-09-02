"""Replaces a fixed, hand-calibrated ramp_ceiling_w_per_s constant with a value that tracks
the fleet's OWN recently observed ramp distribution -- so share_power(c) = ramp(c)/ceiling
self-normalizes to whatever arrival-rate/prompt-length regime is CURRENTLY happening,
instead of assuming one offline burst-test calibration (450 W/s, from one specific
24-concurrent-45k-char-prefill burst against one idle replica) generalizes to every
workload. Floored at that original calibrated value, not free-floating in both directions:
a deliberately one-directional, conservative design that can only ever RELAX the ceiling for
regimes heavier/more volatile than what was originally calibrated for, never tighten it
below the floor. (A fully two-directional design was considered and rejected: letting the
ceiling shrink to match a lighter regime's natural scale would make share_power MORE
sensitive to near-idle measurement noise, not less -- plausibly the same failure mode
already diagnosed for lmetric_power's linear penalty reacting to noise-level differences
below the ceiling.)"""
from collections import deque

DEFAULT_FLOOR_W_PER_S = 450.0
DEFAULT_WINDOW_SIZE = 720  # ~60s at 6 replicas x 500ms poll cadence
DEFAULT_PERCENTILE = 0.99


class AdaptiveRampCeiling:
    def __init__(self, floor_w_per_s: float = DEFAULT_FLOOR_W_PER_S,
                 window_size: int = DEFAULT_WINDOW_SIZE, percentile: float = DEFAULT_PERCENTILE):
        self.floor_w_per_s = floor_w_per_s
        self.percentile = percentile
        self._window = deque(maxlen=window_size)

    def observe(self, ramp_rate_w_per_s: float) -> None:
        """Record a new ramp observation. Clamped to non-negative, matching
        share_power(c)'s own clamp -- a falling ramp is never a pressure signal, so it must
        not be able to pull the ceiling down either."""
        self._window.append(max(ramp_rate_w_per_s, 0.0))

    def observe_round(self, ramp_rates_w_per_s: list) -> None:
        """Records one decision round's fleet-wide ramp readings together, but SKIPS the
        WHOLE round if 2+ replicas are simultaneously elevated (> floor) -- a concentration
        event, which must not inflate the very ceiling meant to guard against it. An
        isolated round (0 or 1 elevated) is benign individual variability, recorded
        normally, same as observe().

        Fixes the self-defeating loop diagnosed in plain observe()-based calibration: since
        "concentration" (multiple replicas elevated at once) is exactly the condition that
        would otherwise fill the window with elevated values, a naive rolling percentile
        raises the ceiling most aggressively DURING a sustained multi-replica pressure
        episode -- diluting protection right when it matters most. This method keeps the
        per-candidate scoring rule itself unchanged (still a single continuous ratio, still
        separable, Lemma 1 still applies) -- only the calibration bookkeeping needs
        fleet-wide visibility, not the routing decision."""
        clamped = [max(r, 0.0) for r in ramp_rates_w_per_s]
        elevated_count = sum(1 for r in clamped if r > self.floor_w_per_s)
        if elevated_count >= 2:
            return
        for r in clamped:
            self._window.append(r)

    def ceiling(self) -> float:
        if not self._window:
            return self.floor_w_per_s
        sorted_vals = sorted(self._window)
        idx = min(int(len(sorted_vals) * self.percentile), len(sorted_vals) - 1)
        return max(self.floor_w_per_s, sorted_vals[idx])
