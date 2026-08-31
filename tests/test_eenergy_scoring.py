import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from scoring import (Candidate, pick_round_robin, pick_lmetric, dominant_share, pick_drf,  # noqa: E402
                      pick_p2c_whale, pick_whale_argmin, share_power, pick_constrained_lmetric,
                      dominant_share_vector)


def _cand(replica_id, new_tokens=0, in_flight_after=1, token_budget=100,
           max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0,
           active_whale_count_after=0):
    return Candidate(replica_id, new_tokens, in_flight_after, token_budget,
                      max_num_seqs, ramp_rate_w_per_s, ramp_ceiling_w_per_s,
                      active_whale_count_after)


class _FakeRng:
    """Deterministic stand-in for random.Random -- .sample() always returns a fixed,
    caller-specified pair regardless of input, so pick_p2c_whale's comparison logic can be
    tested without depending on real randomness."""
    def __init__(self, fixed_sample):
        self.fixed_sample = fixed_sample

    def sample(self, population, k):
        return self.fixed_sample


def test_round_robin_cycles_through_candidates_in_order():
    cands = [_cand("r0"), _cand("r1"), _cand("r2")]
    chosen, idx = pick_round_robin(cands, last_index=-1)
    assert (chosen, idx) == ("r0", 0)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r1", 1)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r2", 2)
    chosen, idx = pick_round_robin(cands, last_index=idx)
    assert (chosen, idx) == ("r0", 0)  # wraps around


def test_lmetric_picks_minimum_new_tokens_times_in_flight():
    cands = [
        _cand("r0", new_tokens=1000, in_flight_after=5),   # score 5000
        _cand("r1", new_tokens=10, in_flight_after=2),     # score 20 (best)
        _cand("r2", new_tokens=50, in_flight_after=50),    # score 2500
    ]
    assert pick_lmetric(cands) == "r1"


def test_dominant_share_is_max_of_three_normalized_shares():
    # compute-dominant: 80/100=0.8, load 1/10=0.1, power 0/10=0.0
    c = _cand("r0", new_tokens=80, in_flight_after=1, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)
    assert dominant_share(c) == 0.8


def test_dominant_share_negative_ramp_does_not_count_as_pressure():
    c = _cand("r0", new_tokens=0, in_flight_after=0, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=-50.0, ramp_ceiling_w_per_s=10.0)
    assert dominant_share(c) == 0.0


def test_drf_picks_replica_with_lowest_dominant_share_even_if_worse_on_other_axes():
    # r0 is best on compute+load but its power ramp is already near the ceiling
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=95.0, ramp_ceiling_w_per_s=100.0)  # dom=0.95
    # r1 is worse on compute+load but has power headroom
    r1 = _cand("r1", new_tokens=50, in_flight_after=5, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # dom=0.5
    assert pick_drf([r0, r1]) == "r1"
    # for comparison: LMETRIC (power-blind) would have picked r0 here
    assert pick_lmetric([r0, r1]) == "r0"


def test_pick_functions_raise_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_lmetric([])
    with pytest.raises(ValueError):
        pick_drf([])
    with pytest.raises(ValueError):
        pick_round_robin([], -1)


def test_dominant_share_vector_is_sorted_descending_and_first_element_matches_dominant_share():
    c = _cand("r0", new_tokens=80, in_flight_after=1, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)
    vec = dominant_share_vector(c)
    assert vec == (0.8, 0.1, 0.0)
    assert vec[0] == dominant_share(c)


def test_drf_lexicographic_tiebreak_uses_load_when_dominant_compute_share_ties():
    """Regression test for the 2026-08-31 diagnosed DRF tail-latency bug: an early,
    un-cacheable whale makes Share_compute identical AND dominant across every candidate
    (much larger than load/power at that point) -- the OLD scalar-max dominant_share ties
    all three candidates and falls back to an arbitrary rotation, ignoring real load
    differences. Proper (lexicographic) DRF must pick the lowest-load candidate among the
    tied-dominant ones instead."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=8, max_num_seqs=10)  # compute=0.9 (dom), load=0.8
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=2, max_num_seqs=10)  # compute=0.9 (dom), load=0.2 (least loaded)
    r2 = _cand("r2", new_tokens=90, token_budget=100, in_flight_after=5, max_num_seqs=10)  # compute=0.9 (dom), load=0.5
    assert pick_drf([r0, r1, r2]) == "r1"


def test_drf_tie_breaking_rotates_instead_of_always_picking_first_candidate():
    """Regression test for the 2026-08-31 load-imbalance bug: an un-cacheable workload makes
    every fresh candidate's dominant share identical (e.g. all-zero), and an unrotated
    min() would then always return candidates[0] -- piling every request onto one replica.
    tie_start must rotate which candidate wins the tie."""
    tied = [_cand("r0"), _cand("r1"), _cand("r2")]  # all identical -> dominant_share ties at 0.0
    assert pick_drf(tied, tie_start=0) == "r0"
    assert pick_drf(tied, tie_start=1) == "r1"
    assert pick_drf(tied, tie_start=2) == "r2"
    assert pick_drf(tied, tie_start=3) == "r0"  # wraps


def test_lmetric_tie_breaking_rotates_instead_of_always_picking_first_candidate():
    tied = [_cand("r0"), _cand("r1"), _cand("r2")]  # all score 0 -> tie
    assert pick_lmetric(tied, tie_start=0) == "r0"
    assert pick_lmetric(tied, tie_start=1) == "r1"
    assert pick_lmetric(tied, tie_start=2) == "r2"


def test_tie_breaking_does_not_override_a_genuine_winner():
    # r1 is strictly best regardless of rotation -- tie_start must not distort real decisions
    cands = [_cand("r0", new_tokens=90), _cand("r1", new_tokens=1), _cand("r2", new_tokens=50)]
    for start in range(3):
        assert pick_drf(cands, tie_start=start) == "r1"
        assert pick_lmetric(cands, tie_start=start) == "r1"


def test_p2c_whale_non_whale_request_delegates_to_lmetric():
    # whale-count would favor r0 (0 active whales) but LMETRIC clearly favors r1 -- a
    # non-whale request must ignore whale count entirely and match plain LMETRIC.
    r0 = _cand("r0", new_tokens=1000, in_flight_after=5, active_whale_count_after=1)
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, active_whale_count_after=5)
    assert pick_p2c_whale([r0, r1], is_whale=False, rng=_FakeRng([r0, r1])) == "r1"
    assert pick_p2c_whale([r0, r1], is_whale=False, rng=_FakeRng([r0, r1])) == pick_lmetric([r0, r1])


def test_p2c_whale_picks_lower_whale_count_among_the_two_sampled():
    # LMETRIC score would favor r0 here (lower new_tokens*in_flight) -- a whale request must
    # ignore LMETRIC entirely and use whichever of the SAMPLED pair has fewer active whales.
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, active_whale_count_after=3)
    r1 = _cand("r1", new_tokens=100, in_flight_after=10, active_whale_count_after=0)
    assert pick_p2c_whale([r0, r1], is_whale=True, rng=_FakeRng([r0, r1])) == "r1"


def test_p2c_whale_only_compares_the_two_sampled_candidates_not_the_whole_pool():
    # r2 has the global-minimum whale count, but the fake rng only samples r0/r1 -- the
    # decision must be confined to the sampled pair (the actual point of power-of-two-choices:
    # it's O(1) work per decision, not a full argmin over every replica).
    r0 = _cand("r0", active_whale_count_after=2)
    r1 = _cand("r1", active_whale_count_after=1)
    r2 = _cand("r2", active_whale_count_after=0)
    assert pick_p2c_whale([r0, r1, r2], is_whale=True, rng=_FakeRng([r0, r1])) == "r1"


def test_p2c_whale_single_candidate_returns_it_without_sampling():
    r0 = _cand("r0", active_whale_count_after=7)
    assert pick_p2c_whale([r0], is_whale=True, rng=_FakeRng([])) == "r0"


def test_p2c_whale_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_p2c_whale([], is_whale=True, rng=_FakeRng([]))
    with pytest.raises(ValueError):
        pick_p2c_whale([], is_whale=False, rng=_FakeRng([]))


def test_whale_argmin_non_whale_request_delegates_to_lmetric():
    r0 = _cand("r0", new_tokens=1000, in_flight_after=5, active_whale_count_after=1)
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, active_whale_count_after=5)
    assert pick_whale_argmin([r0, r1], is_whale=False) == pick_lmetric([r0, r1])


def test_whale_argmin_picks_global_minimum_whale_count_not_just_two_sampled():
    # unlike pick_p2c_whale, this must see r2's global-minimum whale count even though it's
    # neither "sampled" -- the whole point of the ablation is full visibility, no sampling.
    r0 = _cand("r0", active_whale_count_after=2)
    r1 = _cand("r1", active_whale_count_after=1)
    r2 = _cand("r2", active_whale_count_after=0)
    assert pick_whale_argmin([r0, r1, r2], is_whale=True) == "r2"


def test_whale_argmin_ignores_lmetric_score_for_whale_requests():
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, active_whale_count_after=3)
    r1 = _cand("r1", new_tokens=100, in_flight_after=10, active_whale_count_after=0)
    assert pick_whale_argmin([r0, r1], is_whale=True) == "r1"


def test_whale_argmin_tie_breaking_rotates():
    tied = [_cand("r0"), _cand("r1"), _cand("r2")]  # all tied at 0 active whales
    assert pick_whale_argmin(tied, is_whale=True, tie_start=0) == "r0"
    assert pick_whale_argmin(tied, is_whale=True, tie_start=1) == "r1"
    assert pick_whale_argmin(tied, is_whale=True, tie_start=2) == "r2"


def test_whale_argmin_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_whale_argmin([], is_whale=True)
    with pytest.raises(ValueError):
        pick_whale_argmin([], is_whale=False)


def test_share_power_negative_ramp_does_not_count_as_pressure():
    c = _cand("r0", ramp_rate_w_per_s=-50.0, ramp_ceiling_w_per_s=10.0)
    assert share_power(c) == 0.0


def test_share_power_matches_dominant_shares_power_term():
    c = _cand("r0", ramp_rate_w_per_s=8.0, ramp_ceiling_w_per_s=10.0)
    assert share_power(c) == 0.8


def test_constrained_lmetric_matches_plain_lmetric_when_nothing_is_over_ceiling():
    # r0 has the best LMETRIC score AND is within its power ceiling -- must win, same as
    # plain LMETRIC would pick, since the constraint isn't binding for anyone here.
    r0 = _cand("r0", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=5.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    assert pick_constrained_lmetric([r0, r1]) == pick_lmetric([r0, r1]) == "r0"


def test_constrained_lmetric_excludes_a_candidate_over_ceiling_even_with_the_best_lmetric_score():
    # r0 would win on pure LMETRIC score, but it's over its ramp ceiling (share_power > 1.0)
    # -- the constraint must exclude it, leaving r1 as the only feasible choice.
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    assert pick_lmetric([r0, r1]) == "r0"  # sanity: plain LMETRIC would pick r0
    assert pick_constrained_lmetric([r0, r1]) == "r1"


def test_constrained_lmetric_exactly_at_ceiling_counts_as_feasible():
    # share_power == 1.0 exactly (not > 1.0) must still be feasible -- the constraint is
    # "not exceeding" the ceiling, not "strictly below" it.
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=100.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    assert pick_constrained_lmetric([r0, r1]) == "r0"


def test_constrained_lmetric_falls_back_to_least_infeasible_when_nothing_is_feasible():
    # every candidate exceeds its ceiling -- must pick whichever is LEAST over (lowest
    # share_power), not silently fall back to plain LMETRIC (which would defeat the whole
    # point of the constraint in exactly the moment it matters most).
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=500.0, ramp_ceiling_w_per_s=100.0)  # share=5.0
    r1 = _cand("r1", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # share=1.5
    assert pick_lmetric([r0, r1]) == "r0"  # sanity: plain LMETRIC would pick r0 (bad choice here)
    assert pick_constrained_lmetric([r0, r1]) == "r1"  # least-bad on power, not LMETRIC's pick


def test_constrained_lmetric_tie_breaking_rotates():
    tied = [_cand("r0"), _cand("r1"), _cand("r2")]  # all tied, all feasible (ramp=0)
    assert pick_constrained_lmetric(tied, tie_start=0) == "r0"
    assert pick_constrained_lmetric(tied, tie_start=1) == "r1"
    assert pick_constrained_lmetric(tied, tie_start=2) == "r2"


def test_constrained_lmetric_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_constrained_lmetric([])


def test_p2c_whale_with_real_rng_distributes_across_replicas():
    """Not a fake-rng unit test -- exercises the actual random.Random path end to end to
    confirm repeated whale routing under tied whale counts doesn't collapse onto one replica
    (the same class of bug fixed for pick_lmetric/pick_drf, this time by sampling itself
    rather than a rotation cursor)."""
    import random
    cands = [_cand("r0"), _cand("r1"), _cand("r2")]  # all tied at 0 active whales
    rng = random.Random(42)
    chosen = {pick_p2c_whale(cands, is_whale=True, rng=rng) for _ in range(30)}
    assert chosen == {"r0", "r1", "r2"}
