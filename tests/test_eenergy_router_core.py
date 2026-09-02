import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from router_core import Router, WHALE_TOKEN_THRESHOLD  # noqa: E402
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


class _FirstKRng:
    """Deterministic stand-in for random.Random -- .sample() always returns the first k
    elements in the order given (Router builds candidates in replica_states order), so
    router_core-level tests can reason about which replica gets sampled without needing to
    pre-construct the Candidate objects Router builds internally."""
    def sample(self, population, k):
        return population[:k]


def _states():
    cfgs = [
        ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
        ReplicaConfig(replica_id="r1", host="h", port=2, gpu_index=1,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
    ]
    return [ReplicaState(config=c) for c in cfgs]


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        Router(_states(), policy="not_a_real_policy")


def test_round_robin_alternates_across_two_calls():
    r = Router(_states(), policy="round_robin")
    first = r.route(token_ids=[1, 2, 3])
    second = r.route(token_ids=[4, 5, 6])
    assert {first, second} == {"r0", "r1"}
    assert first != second


def test_lmetric_prefers_replica_with_cached_prefix():
    states = _states()
    r = Router(states, policy="lmetric")
    long_prompt = list(range(64))  # 4 full 16-token blocks
    r.route(long_prompt)  # both replicas start empty/idle; tie_start=0 picks first candidate (r0)
    r.complete("r0")
    # route the SAME prompt again: r0 now has it cached (0 new tokens), r1 doesn't
    second = r.route(long_prompt)
    assert second == "r0"


def test_drf_routes_away_from_replica_with_high_ramp_even_if_cache_favors_it():
    states = _states()
    states[0].cached_block_hashes = set()  # r0: no cache advantage
    states[0].ramp_rate_w_per_s = 95.0     # but r0 is near its ramp ceiling (dom share 0.95)
    states[1].ramp_rate_w_per_s = 0.0      # r1 has full power headroom
    r = Router(states, policy="drf")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_drf_power_tiebreak_routes_away_from_replica_with_high_ramp_even_if_cache_favors_it():
    """Integration-level smoke test: drf_power_tiebreak's primary criterion (route to lowest
    dominant share) is unchanged from plain drf -- when shares don't tie, the two must agree.
    Tie-break-specific behavior is covered at the scoring-function level
    (test_eenergy_scoring.py)."""
    states = _states()
    states[0].cached_block_hashes = set()  # r0: no cache advantage
    states[0].ramp_rate_w_per_s = 95.0     # but r0 is near its ramp ceiling (dom share 0.95)
    states[1].ramp_rate_w_per_s = 0.0      # r1 has full power headroom
    r = Router(states, policy="drf_power_tiebreak")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_drf_coincidence_tiebreak_ties_below_ceiling_unlike_power_tiebreak():
    """Real behavioral difference from drf_power_tiebreak, found while writing this test
    (not designed in advance): share_power_coincidence is threshold-gated at share_power >
    1.0 ("is this replica ramping right now"), not continuous like share_power(c) -- so a
    replica at 0.95 (close to its ceiling but not over it) scores IDENTICALLY to one at
    0.0 (idle). drf_power_tiebreak's continuous share correctly steers away from the 0.95
    replica here; drf_coincidence_tiebreak cannot see the difference below the hard
    threshold, both candidates tie on every dimension, and the tie falls through to
    rotation order. This is the coincidence share's known trade-off: it targets the
    fleet-aggregate coincidence event precisely, at the cost of sub-ceiling magnitude
    sensitivity -- documented here rather than silently patched around."""
    states = _states()
    states[0].cached_block_hashes = set()  # r0: no cache advantage
    states[0].ramp_rate_w_per_s = 95.0     # close to ceiling but NOT over it (share_power=0.95)
    states[1].ramp_rate_w_per_s = 0.0      # r1 has full power headroom

    states_old = _states()
    states_old[0].cached_block_hashes = set()
    states_old[0].ramp_rate_w_per_s = 95.0
    states_old[1].ramp_rate_w_per_s = 0.0
    r_old = Router(states_old, policy="drf_power_tiebreak")
    assert r_old.route(token_ids=[1, 2, 3]) == "r1"  # continuous share: correctly avoids r0

    r_new = Router(states, policy="drf_coincidence_tiebreak")
    assert r_new.route(token_ids=[1, 2, 3]) == "r0"  # thresholded share: can't tell 0.95 from 0.0,
                                                       # ties fall through to rotation (r0 first)


def test_drf_coincidence_tiebreak_prefers_already_ramping_replica_over_triggering_a_new_one():
    """The actual hypothesis under test, at integration level: r0 is ALREADY over its ramp
    ceiling, r1 is quiet, and both tie on compute/load. drf_power_tiebreak (per-replica
    power share) would route to r1 to avoid r0's pressure -- but that TRIGGERS a brand new
    simultaneous ramp on r1. drf_coincidence_tiebreak (fleet-aggregate share) should instead
    prefer piling onto r0, since that creates no NEW coincidence event."""
    states = _states()
    states[0].ramp_rate_w_per_s = 150.0  # r0: already ramping (over 100.0 ceiling)
    states[1].ramp_rate_w_per_s = 0.0    # r1: quiet

    states_old = _states()
    states_old[0].ramp_rate_w_per_s = 150.0
    states_old[1].ramp_rate_w_per_s = 0.0
    r_old = Router(states_old, policy="drf_power_tiebreak")
    assert r_old.route(token_ids=[1, 2, 3]) == "r1"  # old share: avoids the ramping replica

    r_new = Router(states, policy="drf_coincidence_tiebreak")
    assert r_new.route(token_ids=[1, 2, 3]) == "r0"  # new share: prefers it instead


def test_drf_peak_power_tiebreak_routes_away_from_replica_near_its_power_ceiling():
    """Integration-level analog of drf_power_tiebreak's own test, using current power LEVEL
    (last_power_w) instead of ramp rate: r0 is drawing nearly its full hardware power limit,
    r1 has full headroom. Primary criterion (route to lowest dominant share) must pick r1."""
    states = _states()
    states[0].cached_block_hashes = set()  # r0: no cache advantage
    states[0].last_power_w = 440.0         # r0: near its 450W hardware limit (dom share ~0.98)
    states[1].last_power_w = 0.0           # r1: full power headroom
    r = Router(states, policy="drf_peak_power_tiebreak")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_drf_power_tiebreak_p2c_picks_better_of_the_sampled_pair():
    """Integration-level smoke test: with a deterministic RNG forcing the sample, the
    router must pick whichever sampled candidate has the lower drf_power_tiebreak score."""
    states = _states()  # only 2 replicas, so the "sample" is just [r0, r1] regardless
    states[0].ramp_rate_w_per_s = 95.0  # r0: near ceiling (dom share 0.95)
    states[1].ramp_rate_w_per_s = 0.0   # r1: full headroom
    r = Router(states, policy="drf_power_tiebreak_p2c", rng=_FirstKRng())
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_drf_power_tiebreak_adaptive_starts_at_the_450_floor():
    states = _states()  # ramp_ceiling_w_per_s=100.0 in config -- irrelevant to this policy,
                         # which uses the adaptive tracker's own floor instead
    r = Router(states, policy="drf_power_tiebreak_adaptive")
    r.route(token_ids=[1, 2, 3])
    assert r.adaptive_ceiling.ceiling() == 450.0  # nothing elevated observed yet


def test_drf_power_tiebreak_adaptive_relaxes_the_ceiling_once_the_regime_is_heavier():
    """After repeatedly observing a heavier-than-450 regime (both replicas consistently
    ramping around 900 W/s), the adaptive ceiling rises well above the static floor -- a
    replica at 900 W/s stops looking maximally pressured once that's normal for the CURRENT
    regime, unlike the fixed-ceiling drf_power_tiebreak, which would treat it as ~2x over
    ceiling forever regardless of how the workload has actually been behaving."""
    states = _states()
    r = Router(states, policy="drf_power_tiebreak_adaptive")
    for _ in range(50):
        states[0].ramp_rate_w_per_s = 900.0
        states[1].ramp_rate_w_per_s = 900.0
        r.route(token_ids=[1, 2, 3])
    assert r.adaptive_ceiling.ceiling() == 900.0


def test_drf_power_tiebreak_adaptive_isolated_starts_at_the_450_floor():
    states = _states()
    r = Router(states, policy="drf_power_tiebreak_adaptive_isolated")
    r.route(token_ids=[1, 2, 3])
    assert r.adaptive_ceiling_isolated.ceiling() == 450.0  # nothing elevated observed yet


def test_drf_power_tiebreak_adaptive_isolated_does_not_relax_under_simultaneous_pressure():
    """Direct contrast with drf_power_tiebreak_adaptive's own test: the EXACT SAME setup
    (both replicas consistently at 900 W/s every round -- a sustained, simultaneous
    concentration episode) that makes the naive adaptive ceiling climb to 900 must leave
    THIS ceiling pinned at the floor throughout, since every round has 2 elevated replicas
    and gets excluded from calibration entirely."""
    states = _states()
    r = Router(states, policy="drf_power_tiebreak_adaptive_isolated")
    for _ in range(50):
        states[0].ramp_rate_w_per_s = 900.0
        states[1].ramp_rate_w_per_s = 900.0
        r.route(token_ids=[1, 2, 3])
    assert r.adaptive_ceiling_isolated.ceiling() == 450.0  # still the floor -- unlike the naive version


def test_drf_power_tiebreak_adaptive_isolated_still_relaxes_for_genuinely_isolated_pressure():
    """The fix only excludes CONCENTRATION, not all adaptation: if only ONE replica is ever
    elevated at a time (never simultaneously with the other), the ceiling still climbs,
    matching the original motivation (track the regime) for the case that's actually safe
    to adapt to."""
    states = _states()
    r = Router(states, policy="drf_power_tiebreak_adaptive_isolated")
    for _ in range(50):
        states[0].ramp_rate_w_per_s = 900.0
        states[1].ramp_rate_w_per_s = 0.0  # only r0 ever elevated -- isolated every round
        r.route(token_ids=[1, 2, 3])
    assert r.adaptive_ceiling_isolated.ceiling() == 900.0


def test_compute_only_prefers_replica_with_cached_prefix_ignoring_load():
    """Integration-level analog of the lmetric cache-preference test: compute_only must also
    steer toward a cache hit (fewer new tokens), and must NOT be swayed off it by load."""
    states = _states()
    r = Router(states, policy="compute_only")
    long_prompt = list(range(64))  # 4 full 16-token blocks
    r.route(long_prompt)  # both replicas start empty/idle; tie_start=0 picks first candidate (r0)
    r.complete("r0")
    # route the SAME prompt again: r0 now has it cached (0 new tokens), r1 doesn't
    second = r.route(long_prompt)
    assert second == "r0"


def test_drf_power_tiebreak_full_matches_drf_power_tiebreak_when_only_power_differs():
    """Integration-level check of the near-free-fix claim: with load equal and only power
    differing, (D, power, load) alone already distinguishes the two candidates, so appending
    compute as a fourth tie-break coordinate must not change the routed replica -- confirmed
    through the full Router policy-dispatch path, not just the pure scoring function."""
    def _pressured_states():
        states = _states()
        states[0].ramp_rate_w_per_s = 80.0  # r0: power=0.8
        states[1].ramp_rate_w_per_s = 0.0   # r1: power=0.0 (lower -- both rules should pick this)
        return states

    r_old = Router(_pressured_states(), policy="drf_power_tiebreak")
    r_full = Router(_pressured_states(), policy="drf_power_tiebreak_full")
    assert r_old.route(token_ids=[1, 2, 3]) == r_full.route(token_ids=[1, 2, 3]) == "r1"


def test_lmetric_power_avoids_pressured_replica_even_when_raw_lmetric_score_ties():
    """Integration-level smoke test mirroring the scoring-level test: two candidates with
    identical raw new_tokens*in_flight_after but different power pressure -- lmetric_power
    must differentiate them via the continuous penalty, with no separate tie-break rule."""
    states = _states()  # ramp_ceiling_w_per_s=100.0 (see _states())
    states[0].ramp_rate_w_per_s = 80.0  # r0: power=0.8
    states[1].ramp_rate_w_per_s = 0.0   # r1: power=0.0
    r = Router(states, policy="lmetric_power")
    r.load_tracker.on_dispatch("r0")  # make BS identical: both replicas at in_flight_after=1
    r.load_tracker.on_dispatch("r1")  # (0 in-flight before this route() call would be +1 each)
    chosen = r.route(token_ids=[1] * 10)
    assert chosen == "r1"


def test_lmetric_power_convex_avoids_pressured_replica_even_when_raw_lmetric_score_ties():
    states = _states()  # ramp_ceiling_w_per_s=100.0 (see _states())
    states[0].ramp_rate_w_per_s = 80.0  # r0: power=0.8
    states[1].ramp_rate_w_per_s = 0.0   # r1: power=0.0
    r = Router(states, policy="lmetric_power_convex")
    r.load_tracker.on_dispatch("r0")
    r.load_tracker.on_dispatch("r1")
    chosen = r.route(token_ids=[1] * 10)
    assert chosen == "r1"


def test_whale_argmin_power_switch_follows_drf_power_tiebreak_when_pressured():
    """Integration-level smoke test: pressured mode routes away from the over-ceiling
    replica even though it would otherwise win on whale-count/LMETRIC grounds."""
    states = _states()
    states[0].cached_block_hashes = set()
    states[0].ramp_rate_w_per_s = 150.0  # r0: over its ramp ceiling (share_power=1.5)
    states[1].ramp_rate_w_per_s = 0.0    # r1: full power headroom
    r = Router(states, policy="whale_argmin_power_switch")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_whale_argmin_power_switch_follows_whale_argmin_when_not_pressured():
    """Integration-level smoke test: unpressured mode, whale request -- fewest active
    whales wins, same as plain whale_argmin."""
    states = _states()
    whale_tokens = list(range(WHALE_TOKEN_THRESHOLD + 1))
    r = Router(states, policy="whale_argmin_power_switch")
    r.route(whale_tokens)          # r0 gets 1 active whale (tie_start=0)
    second = r.route(whale_tokens)
    assert second == "r1"          # r1 has 0 active whales, wins


def test_constrained_lmetric_excludes_replica_over_ramp_ceiling_even_with_cache_advantage():
    states = _states()
    states[0].cached_block_hashes = set()
    states[0].ramp_rate_w_per_s = 150.0  # r0: over its ramp ceiling (share_power=1.5)
    states[1].ramp_rate_w_per_s = 0.0    # r1: full power headroom
    r = Router(states, policy="constrained_lmetric")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"


def test_constrained_lmetric_matches_plain_lmetric_when_no_one_is_over_ceiling():
    states_a = _states()
    r_a = Router(states_a, policy="constrained_lmetric")
    long_prompt = list(range(64))
    r_a.route(long_prompt)
    r_a.complete("r0")
    chosen_a = r_a.route(long_prompt)

    states_b = _states()
    r_b = Router(states_b, policy="lmetric")
    r_b.route(long_prompt)
    r_b.complete("r0")
    chosen_b = r_b.route(long_prompt)

    assert chosen_a == chosen_b == "r0"  # cache-affinity wins, identical to plain LMETRIC


def test_route_then_complete_updates_load_tracker_round_trip():
    r = Router(_states(), policy="round_robin")
    chosen = r.route(token_ids=[1])
    assert r.load_tracker.in_flight(chosen) == 1
    r.complete(chosen)
    assert r.load_tracker.in_flight(chosen) == 0


def test_whale_token_threshold_is_a_positive_admission_time_cutoff():
    assert WHALE_TOKEN_THRESHOLD > 0


def test_p2c_whale_non_whale_request_matches_plain_lmetric_behavior():
    long_prompt = list(range(64))  # 4 full 16-token blocks, but well under the whale threshold
    states_p2c = _states()
    r_p2c = Router(states_p2c, policy="p2c_whale", rng=_FirstKRng())
    r_p2c.route(long_prompt)
    r_p2c.complete("r0")
    chosen_p2c = r_p2c.route(long_prompt)

    states_lmetric = _states()
    r_lmetric = Router(states_lmetric, policy="lmetric")
    r_lmetric.route(long_prompt)
    r_lmetric.complete("r0")
    chosen_lmetric = r_lmetric.route(long_prompt)

    assert chosen_p2c == chosen_lmetric == "r0"  # cache-affinity wins, same as plain LMETRIC


def test_p2c_whale_whale_request_uses_active_whale_count_not_lmetric_score():
    states = _states()
    r = Router(states, policy="p2c_whale", rng=_FirstKRng())
    whale_tokens = list(range(WHALE_TOKEN_THRESHOLD + 1))

    # give r0 an active whale already (worse whale count) but a cache advantage LMETRIC would
    # normally prefer (0 new tokens) -- P2C-whale must ignore the cache advantage entirely
    r.route(whale_tokens)  # r0 wins the first (tied) dispatch under _FirstKRng, gets cached
    second = r.route(whale_tokens)  # r0: 1 active whale + full cache hit; r1: 0 active whales
    assert second == "r1"


def test_p2c_whale_complete_decrements_whale_tracker():
    states = _states()
    r = Router(states, policy="p2c_whale", rng=_FirstKRng())
    whale_tokens = list(range(WHALE_TOKEN_THRESHOLD + 1))
    chosen = r.route(whale_tokens)
    assert r.whale_tracker.active_whale_count(chosen) == 1
    r.complete(chosen, is_whale=True)
    assert r.whale_tracker.active_whale_count(chosen) == 0


def test_whale_argmin_sees_global_minimum_whale_count_not_just_two_sampled():
    """Ablation for the coincidence-frequency question: whale_argmin has no rng at all --
    if it still doesn't beat DRF's coincidence figure once measured on hardware, the gap
    isn't a sampling-size artifact."""
    cfgs = [
        ReplicaConfig(replica_id="r0", host="h", port=1, gpu_index=0,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
        ReplicaConfig(replica_id="r1", host="h", port=2, gpu_index=1,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
        ReplicaConfig(replica_id="r2", host="h", port=3, gpu_index=2,
                       token_budget=1000, max_num_seqs=10, ramp_ceiling_w_per_s=100.0),
    ]
    states = [ReplicaState(config=c) for c in cfgs]
    r = Router(states, policy="whale_argmin")  # no rng needed -- deterministic
    whale_tokens = list(range(WHALE_TOKEN_THRESHOLD + 1))

    # load up r0 and r1 with active whales; r2 stays empty -- global argmin must find r2
    # even though it's not "sampled" (there's no sampling in this policy at all)
    r.route(whale_tokens)  # r0 (tie_start=0)
    second = r.route(whale_tokens)
    assert second != "r0"  # r0 already has 1 active whale, so it's no longer the minimum
    third = r.route(whale_tokens)
    assert third == "r2"  # r2 is the only replica with 0 active whales left


def test_custom_whale_token_threshold_overrides_default():
    """Lets a workload with genuine but smaller size variance than the synthetic
    whale-injection workload (e.g. real BurstGPT traffic, max ~4k tokens vs. the
    13.6-15.5k-token whales this project's default 4000 threshold was calibrated
    against) still engage whale-aware routing at its own natural scale."""
    states = _states()
    r = Router(states, policy="whale_argmin", whale_token_threshold=10)
    above = list(range(11))  # 11 tokens: not a whale under the default 4000, but is one here

    r.route(above)  # r0 gets 1 active whale (tie_start=0)
    chosen = r.route(above)
    assert chosen == "r1"  # whale_argmin avoids r0's active whale


def test_default_whale_token_threshold_matches_module_constant():
    states = _states()
    r = Router(states, policy="round_robin")
    assert r.whale_token_threshold == WHALE_TOKEN_THRESHOLD


def test_route_exposes_last_is_whale_for_logging():
    r = Router(_states(), policy="round_robin")
    r.route(token_ids=list(range(WHALE_TOKEN_THRESHOLD + 1)))
    assert r.last_is_whale is True
    r.route(token_ids=[1, 2, 3])
    assert r.last_is_whale is False


def test_complete_without_is_whale_stays_backward_compatible():
    r = Router(_states(), policy="round_robin")
    chosen = r.route(token_ids=[1])
    r.complete(chosen)  # no is_whale arg -- must not raise, must not touch whale_tracker
    assert r.whale_tracker.active_whale_count(chosen) == 0


def test_pressure_switch_matches_lmetric_when_fleet_is_not_pressured():
    states_a = _states()
    r_a = Router(states_a, policy="pressure_switch")
    long_prompt = list(range(64))
    r_a.route(long_prompt)
    r_a.complete("r0")
    chosen_a = r_a.route(long_prompt)

    states_b = _states()
    r_b = Router(states_b, policy="lmetric")
    r_b.route(long_prompt)
    r_b.complete("r0")
    chosen_b = r_b.route(long_prompt)

    assert chosen_a == chosen_b == "r0"  # cache-affinity wins, identical to plain LMETRIC


def test_pressure_switch_routes_away_from_over_ceiling_replica_even_with_cache_advantage():
    states = _states()
    states[0].cached_block_hashes = set()
    states[0].ramp_rate_w_per_s = 150.0  # r0: over its ramp ceiling -- fleet is pressured
    states[1].ramp_rate_w_per_s = 0.0    # r1: full power headroom
    r = Router(states, policy="pressure_switch")
    chosen = r.route(token_ids=[1, 2, 3])
    assert chosen == "r1"  # DRF's decision, not LMETRIC's (LMETRIC is blind to power)


def test_bs_source_local_is_the_default_and_matches_prior_behavior():
    """Regression: bs_source defaults to 'local' (load_tracker), byte-identical to the
    Router's behavior before bs_source existed at all."""
    states = _states()
    r = Router(states, policy="lmetric")
    long_prompt = list(range(64))
    r.route(long_prompt)
    r.complete("r0")
    chosen = r.route(long_prompt)
    assert chosen == "r0"  # cache-affinity wins, same as the pre-existing lmetric test


def test_bs_source_telemetry_uses_state_telemetry_bs_not_load_tracker():
    states = _states()
    # r0 has zero dispatch/complete history (load_tracker says 0) but telemetry says it's
    # already busy; r1 is the reverse. bs_source="telemetry" must follow telemetry, not
    # load_tracker, or this test can't distinguish the two sources.
    states[0].telemetry_bs = 20   # heavily loaded per real engine telemetry
    states[1].telemetry_bs = 0    # idle per real engine telemetry
    r = Router(states, policy="lmetric", bs_source="telemetry")
    chosen = r.route(token_ids=[1, 2, 3])  # uncacheable, so new_tokens ties -- only BS decides
    assert chosen == "r1"


def test_bs_source_telemetry_still_updates_load_tracker_and_whale_tracker():
    """load_tracker/whale_tracker keep tracking regardless of bs_source -- just not read for
    the routing decision in telemetry mode. Cheap, and keeps whale bookkeeping consistent."""
    states = _states()
    r = Router(states, policy="lmetric", bs_source="telemetry")
    chosen = r.route(token_ids=[1])
    assert r.load_tracker.in_flight(chosen) == 1
    r.complete(chosen)
    assert r.load_tracker.in_flight(chosen) == 0


def test_route_exposes_the_chosen_candidates_new_tokens_for_logging():
    """Router.route() must expose the P-token value it actually used for the chosen replica
    -- otherwise there's no way to audit, post-hoc, whether the KV$-hit discount was honest
    once real cache hits start occurring (the assignment log wires this up next)."""
    states = _states()
    r = Router(states, policy="lmetric")
    long_prompt = list(range(64))  # 4 full 16-token blocks, all new on a cold replica
    r.route(long_prompt)
    assert r.last_new_tokens == 64  # cold: no cache hit anywhere, full prompt length
    r.complete("r0")
    r.route(long_prompt)  # same prompt again: r0 now has it fully cached
    assert r.last_new_tokens == 0  # full cache hit


def test_drf_distributes_tied_requests_instead_of_piling_onto_one_replica():
    """Regression test for the 2026-08-31 load-imbalance bug: an un-cacheable workload (a
    fresh, never-seen token_ids every call) makes every candidate's dominant share tie at
    0.0 on every single call. Router must distribute those ties across replicas over
    repeated calls, not always dispatch to the same one."""
    states = _states()
    r = Router(states, policy="drf")
    chosen = set()
    for i in range(4):
        rid = r.route(token_ids=[1000 + i])  # distinct, never-cached prompt each time
        chosen.add(rid)
        r.complete(rid)
    assert chosen == {"r0", "r1"}
