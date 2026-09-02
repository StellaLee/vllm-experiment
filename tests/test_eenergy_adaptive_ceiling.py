import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from adaptive_ceiling import AdaptiveRampCeiling  # noqa: E402


def test_empty_window_returns_the_floor():
    c = AdaptiveRampCeiling(floor_w_per_s=450.0)
    assert c.ceiling() == 450.0


def test_ceiling_stays_at_floor_when_observed_regime_is_lighter_than_the_floor():
    """Light-load regime: every observed ramp is well below the floor -- the floor wins,
    so behavior matches the original fixed-ceiling design exactly in this regime (a
    deliberately one-directional design: this can only ever RELAX the ceiling for regimes
    heavier than what was originally calibrated for, never tighten it below the floor)."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=100, percentile=0.99)
    for _ in range(100):
        c.observe(5.0)
    assert c.ceiling() == 450.0


def test_ceiling_adapts_upward_when_the_regime_is_heavier_than_the_floor():
    """Heavy/volatile regime: most samples are far above the floor -- the ceiling should
    track the window's own high percentile instead of staying pinned to the stale floor."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=100, percentile=0.99)
    for _ in range(99):
        c.observe(1000.0)
    c.observe(5000.0)  # the one outlier at the very top of the p99 window
    assert c.ceiling() == 5000.0  # p99 of 100 samples lands on the single highest value


def test_negative_ramp_observations_are_clamped_to_zero():
    """Matches share_power(c)'s own clamp semantics: a falling ramp is never a pressure
    signal, so it must not be able to pull the ceiling down either."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=10, percentile=0.99)
    for _ in range(10):
        c.observe(-999.0)
    assert c.ceiling() == 450.0  # all clamped to 0, floor wins


def test_window_is_bounded_and_forgets_old_observations():
    """Old, no-longer-representative samples must eventually age out -- this is what makes
    the ceiling track the CURRENT regime rather than an average over the router's entire
    lifetime."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=5, percentile=0.99)
    for _ in range(5):
        c.observe(9000.0)  # fills the window with a heavy-regime spike
    assert c.ceiling() == 9000.0
    for _ in range(5):
        c.observe(10.0)  # regime cools down; the spike should fully age out of a size-5 window
    assert c.ceiling() == 450.0  # back to the floor, the old spike is gone


def test_percentile_selects_the_correct_rank_in_a_mixed_window():
    c = AdaptiveRampCeiling(floor_w_per_s=0.0, window_size=10, percentile=0.9)
    for v in [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]:
        c.observe(float(v))
    # 90th percentile (index 9 of 10 sorted values, 0-indexed) is the top value, 1000
    assert c.ceiling() == 1000.0
