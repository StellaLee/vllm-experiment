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


# ---- observe_round: coincidence-filtered calibration (the fix for the self-defeating loop
# diagnosed in drf_power_tiebreak_adaptive -- a sustained multi-replica pressure episode
# must not inflate its own tolerance for pressure) ----

def test_observe_round_records_an_isolated_elevated_reading_normally():
    """Only ONE replica elevated this round -- a benign individual transition, not a
    concentration event. Must be recorded like any ordinary observation."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=10, percentile=0.99)
    c.observe_round([900.0, 10.0, 10.0])
    assert c.ceiling() == 900.0


def test_observe_round_excludes_a_simultaneous_multi_replica_round_entirely():
    """TWO replicas elevated in the SAME round -- a concentration event. None of this
    round's readings (elevated or not) may enter the window, or the ceiling would relax
    exactly when protection matters most."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=10, percentile=0.99)
    c.observe_round([900.0, 800.0, 10.0])
    assert c.ceiling() == 450.0  # excluded entirely -- floor still wins


def test_observe_round_records_a_fully_calm_round_normally():
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=10, percentile=0.99)
    c.observe_round([10.0, 20.0, 5.0])
    assert c.ceiling() == 450.0  # nothing elevated, floor wins (trivially recorded)


def test_observe_round_isolated_rounds_can_still_adapt_the_ceiling_upward_over_time():
    """A sequence of ISOLATED elevated rounds (never 2+ simultaneously) should still let
    the ceiling climb -- the fix only excludes CONCENTRATION events, not all adaptation."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=100, percentile=0.99)
    for _ in range(99):
        c.observe_round([1000.0, 10.0, 10.0])  # isolated each time -- only replica 0 elevated
    c.observe_round([5000.0, 10.0, 10.0])  # isolated outlier at the top
    assert c.ceiling() == 5000.0


def test_observe_round_a_sustained_concentration_episode_never_moves_the_ceiling():
    """The exact failure mode being fixed: many consecutive rounds where 2+ replicas are
    simultaneously elevated (a sustained pressure episode) must leave the ceiling pinned at
    the floor throughout -- not creep upward as the episode continues."""
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=100, percentile=0.99)
    for _ in range(50):
        c.observe_round([900.0, 900.0, 10.0])  # 2 replicas simultaneously elevated, every round
    assert c.ceiling() == 450.0


def test_observe_round_clamps_negative_readings_before_counting_as_elevated():
    c = AdaptiveRampCeiling(floor_w_per_s=450.0, window_size=10, percentile=0.99)
    c.observe_round([900.0, -900.0, 10.0])  # only 1 truly elevated after clamping -- isolated
    assert c.ceiling() == 900.0
