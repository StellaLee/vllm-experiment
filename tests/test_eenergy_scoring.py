import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from scoring import Candidate, pick_round_robin, pick_lmetric, dominant_share, pick_drf  # noqa: E402


def _cand(replica_id, new_tokens=0, in_flight_after=1, token_budget=100,
           max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0):
    return Candidate(replica_id, new_tokens, in_flight_after, token_budget,
                      max_num_seqs, ramp_rate_w_per_s, ramp_ceiling_w_per_s)


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
