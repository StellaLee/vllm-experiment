import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from scoring import (Candidate, pick_round_robin, pick_lmetric, dominant_share, pick_drf,  # noqa: E402
                      pick_p2c_whale, pick_whale_argmin, share_power, pick_constrained_lmetric,
                      dominant_share_vector, pick_pressure_switch, fleet_pressured,
                      dominant_share_vector_power_priority, pick_drf_power_tiebreak,
                      lmetric_power_score, pick_lmetric_power,
                      lmetric_power_convex_score, pick_lmetric_power_convex,
                      pick_whale_argmin_power_switch,
                      share_power_coincidence, dominant_share_coincidence,
                      dominant_share_vector_power_priority_coincidence,
                      pick_drf_coincidence_tiebreak,
                      share_power_level, dominant_share_peak,
                      dominant_share_vector_peak_priority, pick_drf_peak_power_tiebreak,
                      pick_drf_power_tiebreak_p2c, pick_compute_only)


def _cand(replica_id, new_tokens=0, in_flight_after=1, token_budget=100,
           max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0,
           active_whale_count_after=0, power_w=0.0, power_level_ceiling_w=450.0):
    return Candidate(replica_id, new_tokens, in_flight_after, token_budget,
                      max_num_seqs, ramp_rate_w_per_s, ramp_ceiling_w_per_s,
                      active_whale_count_after, power_w, power_level_ceiling_w)


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


def test_compute_only_ignores_load_where_lmetric_would_be_swayed_by_it():
    """Ablation isolating whether lmetric's power-ramp-smoothing benefit (session
    discussion: does Share_compute alone already capture most of a PES-IM-chunking-style
    smoothing effect, with no power telemetry and not even a load term) comes from the
    compute term alone or needs the load term too. r0 has fewer new tokens but is heavily
    loaded; r1 has more new tokens but is idle. lmetric's product can be swayed toward r1 by
    a large enough load gap; compute_only ignores load entirely and must always pick r0."""
    cands = [
        _cand("r0", new_tokens=10, in_flight_after=50),   # lmetric score 500
        _cand("r1", new_tokens=100, in_flight_after=1),   # lmetric score 100 (lmetric picks r1)
    ]
    assert pick_lmetric(cands) == "r1"       # swayed by the load term
    assert pick_compute_only(cands) == "r0"  # load-blind: fewest new tokens wins regardless


def test_compute_only_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_compute_only([])


def test_compute_only_ties_rotate_via_tie_start():
    cands = [_cand("r0", new_tokens=5), _cand("r1", new_tokens=5), _cand("r2", new_tokens=5)]
    assert pick_compute_only(cands, tie_start=0) == "r0"
    assert pick_compute_only(cands, tie_start=1) == "r1"
    assert pick_compute_only(cands, tie_start=2) == "r2"


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


def test_dominant_share_vector_power_priority_puts_power_before_load():
    """dominant_share_vector's tie-break is a pure magnitude sort -- whichever of
    {load, power} happens to be numerically larger wins the tie-break, by accident of scale,
    not because the project actually wants load prioritized over power. This variant fixes
    the tie-break's resource ORDER: (dominant_share, power, load), always -- power gets first
    chance to break a tie regardless of which one is numerically bigger."""
    c = _cand("r0", new_tokens=80, in_flight_after=7, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=3.0, ramp_ceiling_w_per_s=10.0)
    # compute=0.8 (dom), load=0.7, power=0.3 -- magnitude sort would put load (0.7) before
    # power (0.3); resource-priority order always puts power second regardless.
    vec = dominant_share_vector_power_priority(c)
    assert vec == (0.8, 0.3, 0.7)
    assert vec[0] == dominant_share(c)


def test_drf_power_tiebreak_prefers_lower_power_over_lower_load_when_compute_ties():
    """The illustrative case the fix targets: two candidates tied on dominant compute share.
    r0 has MORE load but LESS power pressure; r1 has LESS load but MORE power pressure.
    Magnitude-sort tie-break (plain pick_drf) picks whichever's second-largest share is
    smaller regardless of which resource it is -- here that happens to pick r1 (favors low
    load, ignores that r1 is worse on power). Power-priority tie-break must pick r0 instead
    (favors low power, this project's actual protection target), even though r0 has 3.5x
    r1's load."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=7, max_num_seqs=10,
               ramp_rate_w_per_s=3.0, ramp_ceiling_w_per_s=10.0)  # compute=0.9(dom), load=0.7, power=0.3
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=2, max_num_seqs=10,
               ramp_rate_w_per_s=5.0, ramp_ceiling_w_per_s=10.0)  # compute=0.9(dom), load=0.2, power=0.5
    assert pick_drf([r0, r1]) == "r1"  # existing magnitude-sort tie-break: favors low load
    assert pick_drf_power_tiebreak([r0, r1]) == "r0"  # power-priority tie-break: favors low power


def test_drf_power_tiebreak_matches_plain_drf_when_dominant_shares_dont_tie():
    """Same primary criterion as pick_drf (route to minimum dominant share) -- only the
    tie-break order changes, so when there's no tie the two must agree."""
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=95.0, ramp_ceiling_w_per_s=100.0)  # dom=0.95
    r1 = _cand("r1", new_tokens=50, in_flight_after=5, token_budget=100,
               max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # dom=0.5
    assert pick_drf([r0, r1]) == pick_drf_power_tiebreak([r0, r1]) == "r1"


def test_lmetric_power_score_is_tokens_times_bs_times_one_plus_share_power():
    """Score = new_tokens x in_flight_after x (1 + share_power) -- LMETRIC's own
    multiplicative form (Zhang et al., OSDI'26), extended with a continuous power penalty
    instead of DRF's max()-of-shares (which lets one dimension silence the other two, the
    root cause diagnosed for every DRF tie-break failure mode this session)."""
    c = _cand("r0", new_tokens=10, in_flight_after=5, ramp_rate_w_per_s=5.0, ramp_ceiling_w_per_s=10.0)
    # power = 5/10 = 0.5 -> score = 10 * 5 * 1.5 = 75.0
    assert lmetric_power_score(c) == 75.0


def test_lmetric_power_matches_lmetric_when_no_power_pressure():
    """Under zero ramp everywhere (share_power=0 for every candidate), (1+share_power)=1
    for everyone -- the formula collapses to plain LMETRIC exactly. This is the light-load
    prediction: should inherit LMETRIC's known-good light-load behavior for free, unlike
    drf_power_tiebreak which regressed on the light cache-hit workload."""
    cands = [
        _cand("r0", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
        _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
        _cand("r2", new_tokens=50, in_flight_after=50, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
    ]
    assert pick_lmetric_power(cands) == pick_lmetric(cands) == "r1"


def test_lmetric_power_prefers_lower_power_replica_when_raw_lmetric_score_ties():
    """The illustrative case the design targets: r0 and r1 have IDENTICAL raw
    new_tokens*in_flight_after (100 each), so plain LMETRIC can't tell them apart and ties
    (picks the first by rotation). r0 has real power pressure, r1 doesn't -- the
    multiplicative penalty must differentiate them even though the underlying LMETRIC
    scheduling term never would, without needing any separate tie-break rule at all."""
    r0 = _cand("r0", new_tokens=10, in_flight_after=10, ramp_rate_w_per_s=8.0, ramp_ceiling_w_per_s=10.0)  # power=0.8, score=100*1.8=180
    r1 = _cand("r1", new_tokens=10, in_flight_after=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)  # power=0.0, score=100*1.0=100
    assert pick_lmetric([r0, r1]) == "r0"  # plain LMETRIC ties on raw score, picks first by rotation
    assert pick_lmetric_power([r0, r1]) == "r1"  # power-aware avoids the pressured replica


def test_lmetric_power_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_lmetric_power([])


def test_lmetric_power_convex_score_is_tokens_times_bs_times_one_plus_share_power_squared():
    """Score = new_tokens x in_flight_after x (1 + share_power^2) -- same LMETRIC-style
    multiplicative form as lmetric_power, but a convex penalty instead of linear. Convex
    ramp-cost penalties are the standard convention in power-systems economic dispatch /
    unit commitment literature (stressing a generator near its ramp limit carries
    disproportionate, super-linear cost), and directly targets the oscillation mechanism
    diagnosed for lmetric_power's mean_ramp/duty_cycle regression: a linear penalty reacts to
    noise-level power differences even at low pressure; squaring makes the penalty much
    smaller than linear at low share_power (0.3^2=0.09 vs 0.3) while still growing sharply
    near the ceiling (barely different from linear at share_power=1)."""
    c = _cand("r0", new_tokens=10, in_flight_after=5, ramp_rate_w_per_s=5.0, ramp_ceiling_w_per_s=10.0)
    # power = 0.5 -> power^2 = 0.25 -> score = 10 * 5 * 1.25 = 62.5
    assert lmetric_power_convex_score(c) == 62.5


def test_lmetric_power_convex_penalty_is_smaller_than_linear_below_the_ceiling():
    """The core property the design targets: for any share_power in (0, 1), the convex
    penalty is strictly smaller (less reactive to noise) than the linear one; at the ceiling
    (share_power=1) they coincide; beyond it, convex overtakes linear (steeper punishment for
    genuinely exceeding the calibrated ceiling)."""
    below = _cand("r0", new_tokens=10, in_flight_after=5, ramp_rate_w_per_s=3.0, ramp_ceiling_w_per_s=10.0)  # power=0.3
    assert lmetric_power_convex_score(below) < lmetric_power_score(below)

    at_ceiling = _cand("r0", new_tokens=10, in_flight_after=5, ramp_rate_w_per_s=10.0, ramp_ceiling_w_per_s=10.0)  # power=1.0
    assert lmetric_power_convex_score(at_ceiling) == lmetric_power_score(at_ceiling)

    over = _cand("r0", new_tokens=10, in_flight_after=5, ramp_rate_w_per_s=15.0, ramp_ceiling_w_per_s=10.0)  # power=1.5
    assert lmetric_power_convex_score(over) > lmetric_power_score(over)


def test_lmetric_power_convex_matches_lmetric_when_no_power_pressure():
    cands = [
        _cand("r0", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
        _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
        _cand("r2", new_tokens=50, in_flight_after=50, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0),
    ]
    assert pick_lmetric_power_convex(cands) == pick_lmetric(cands) == "r1"


def test_lmetric_power_convex_prefers_lower_power_replica_when_raw_lmetric_score_ties():
    r0 = _cand("r0", new_tokens=10, in_flight_after=10, ramp_rate_w_per_s=8.0, ramp_ceiling_w_per_s=10.0)  # power=0.8
    r1 = _cand("r1", new_tokens=10, in_flight_after=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)  # power=0.0
    assert pick_lmetric_power_convex([r0, r1]) == "r1"


def test_lmetric_power_convex_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_lmetric_power_convex([])


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


def test_fleet_pressured_false_when_no_candidate_over_ceiling():
    r0 = _cand("r0", ramp_rate_w_per_s=50.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", ramp_rate_w_per_s=100.0, ramp_ceiling_w_per_s=100.0)  # exactly at ceiling
    assert fleet_pressured([r0, r1]) is False


def test_fleet_pressured_true_when_any_candidate_over_ceiling():
    r0 = _cand("r0", ramp_rate_w_per_s=50.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # over ceiling
    assert fleet_pressured([r0, r1]) is True


def test_pressure_switch_matches_lmetric_when_fleet_is_not_pressured():
    r0 = _cand("r0", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=50.0, ramp_ceiling_w_per_s=100.0)
    assert pick_pressure_switch([r0, r1]) == pick_lmetric([r0, r1]) == "r1"


def test_pressure_switch_matches_drf_when_fleet_is_pressured():
    # r0 is best on LMETRIC score (lowest new_tokens*in_flight_after) but is ALSO the
    # over-ceiling replica -- DRF's dominant share correctly steers away from it while
    # LMETRIC, blind to power, would pick it anyway. pressure_switch must follow DRF here.
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # over ceiling
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=20.0, ramp_ceiling_w_per_s=100.0)
    assert pick_lmetric([r0, r1]) == "r0"  # sanity: LMETRIC would pick r0
    assert pick_pressure_switch([r0, r1]) == pick_drf([r0, r1]) == "r1"


def test_pressure_switch_tie_breaking_rotates_in_both_regimes():
    tied = [_cand("r0"), _cand("r1"), _cand("r2")]  # all tied, fleet not pressured (ramp=0)
    assert pick_pressure_switch(tied, tie_start=0) == "r0"
    assert pick_pressure_switch(tied, tie_start=1) == "r1"
    assert pick_pressure_switch(tied, tie_start=2) == "r2"


def test_pressure_switch_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_pressure_switch([])


def test_whale_argmin_power_switch_matches_whale_argmin_non_whale_when_not_pressured():
    """Default mode: fleet not pressured (every share_power <= 1.0) -- non-whale request
    delegates through whale_argmin to plain LMETRIC, unchanged."""
    r0 = _cand("r0", new_tokens=1000, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=50.0, ramp_ceiling_w_per_s=100.0)
    assert pick_whale_argmin_power_switch([r0, r1], is_whale=False) == pick_whale_argmin([r0, r1], is_whale=False) == "r1"


def test_whale_argmin_power_switch_matches_whale_argmin_whale_case_when_not_pressured():
    """Default mode, whale request: fewest active_whale_count wins, same as plain
    whale_argmin -- ignores the tied-LMETRIC-favoring candidate the same way whale_argmin
    does."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=1, ramp_rate_w_per_s=0.0,
               ramp_ceiling_w_per_s=100.0, active_whale_count_after=2)
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=1, ramp_rate_w_per_s=0.0,
               ramp_ceiling_w_per_s=100.0, active_whale_count_after=0)
    assert pick_whale_argmin_power_switch([r0, r1], is_whale=True) == pick_whale_argmin([r0, r1], is_whale=True) == "r1"


def test_whale_argmin_power_switch_matches_drf_power_tiebreak_when_fleet_pressured():
    """Pressured mode: switches to drf_power_tiebreak regardless of is_whale -- even a
    non-whale request must follow the power-aware DRF routing while the fleet is pressured,
    the same 'fleet-wide, not per-candidate' semantics pressure_switch already documents."""
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # over ceiling
    r1 = _cand("r1", new_tokens=10, in_flight_after=2, ramp_rate_w_per_s=20.0, ramp_ceiling_w_per_s=100.0)
    assert pick_whale_argmin([r0, r1], is_whale=False) == "r0"  # sanity: whale_argmin (via LMETRIC) would pick r0
    assert pick_whale_argmin_power_switch([r0, r1], is_whale=False) == pick_drf_power_tiebreak([r0, r1]) == "r1"


def test_whale_argmin_power_switch_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_whale_argmin_power_switch([], is_whale=False)


def test_share_power_coincidence_is_zero_when_fleet_fully_calm():
    """A lone replica starting to ramp is not a coincidence -- only 2+ simultaneous ramps
    are. c is not currently ramping and nobody else is either, so routing here would push
    the fleet to exactly 1 ramping replica: still zero coincidence share."""
    r0 = _cand("r0", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    r1 = _cand("r1", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    r2 = _cand("r2", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)
    assert share_power_coincidence(r0, [r0, r1, r2]) == 0.0


def test_share_power_coincidence_penalizes_triggering_a_new_ramp_when_others_already_ramping():
    """r0 is already over ceiling (ramping). Routing to quiet r1 would make it a SECOND
    simultaneously-ramping replica -- a real coincidence event -- while r1 staying quiet
    would not. This is the direct fleet-aggregate signal share_power(c) alone can't see:
    share_power(r1) here is 0.0 (r1's OWN ramp is nil), but routing to r1 still creates a
    new coincidence event because of what r0 is ALREADY doing."""
    r0 = _cand("r0", ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # already ramping
    r1 = _cand("r1", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)    # quiet
    cands = [r0, r1]
    assert share_power(r1) == 0.0  # the OLD per-replica share sees no risk at all here
    assert share_power_coincidence(r1, cands) == 1.0  # resulting_k=2 -> max(2-1,0)/(2-1)=1.0


def test_share_power_coincidence_is_free_to_route_onto_an_already_ramping_replica():
    """Routing MORE load onto a replica that's already ramping doesn't trigger a NEW
    coincidence event -- it was already counted. Counterintuitive vs. share_power(c) (which
    would score this replica as maximally risky), but correct for the fleet-aggregate
    metric: the coincidence count doesn't change whether or not this request lands here."""
    r0 = _cand("r0", ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # already ramping
    r1 = _cand("r1", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)    # quiet
    cands = [r0, r1]
    assert share_power_coincidence(r0, cands) == 0.0  # resulting_k=1 (r0 already counted) -> 0.0
    assert share_power(r0) == 1.5  # the OLD per-replica share scores this as the WORST option


def test_share_power_coincidence_scales_with_fleet_size_and_existing_ramp_count():
    """3 of 6 replicas already ramping; c (r3) is quiet. Routing here makes it the 4th
    simultaneous ramp: max(4-1,0)/(6-1) = 3/5."""
    ramping = [_cand(f"r{i}", ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0) for i in range(3)]
    quiet = [_cand(f"r{i}", ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0) for i in range(3, 6)]
    cands = ramping + quiet
    assert share_power_coincidence(quiet[0], cands) == pytest.approx(3 / 5)


def test_share_power_coincidence_handles_single_candidate_fleet():
    """No other replica can coincide with -- always zero, no division by zero."""
    r0 = _cand("r0", ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)
    assert share_power_coincidence(r0, [r0]) == 0.0


def test_dominant_share_coincidence_uses_coincidence_share_as_the_power_dimension():
    # compute=0.8 (dom), load=0.1; power via coincidence is 0.0 (fleet fully calm)
    c = _cand("r0", new_tokens=80, in_flight_after=1, token_budget=100,
              max_num_seqs=10, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=10.0)
    assert dominant_share_coincidence(c, [c]) == 0.8


def test_drf_coincidence_tiebreak_prefers_the_already_ramping_replica_when_fleet_is_pressured():
    """Direct behavioral contrast with pick_drf_power_tiebreak: when one replica is already
    ramping and the other is quiet, and both otherwise tie on compute/load, the coincidence-
    aware tie-break should PREFER piling onto the already-ramping replica (no new coincidence
    event) -- the opposite of what the per-replica power share does."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=5, max_num_seqs=10,
               ramp_rate_w_per_s=150.0, ramp_ceiling_w_per_s=100.0)  # already ramping
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=5, max_num_seqs=10,
               ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # quiet
    assert pick_drf_power_tiebreak([r0, r1]) == "r1"  # old share: avoids the already-ramping one
    assert pick_drf_coincidence_tiebreak([r0, r1]) == "r0"  # new share: prefers it instead


def test_drf_coincidence_tiebreak_matches_plain_drf_when_fleet_fully_calm():
    """With nobody ramping, share_power_coincidence is 0.0 for every candidate (see
    test_share_power_coincidence_is_zero_when_fleet_fully_calm) -- so the coincidence
    tie-break degenerates to load-only tie-breaking, same as plain pick_drf's lexicographic
    sort would do once power drops out as the smallest, non-discriminating share."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=8, max_num_seqs=10)
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=2, max_num_seqs=10)
    assert pick_drf([r0, r1]) == pick_drf_coincidence_tiebreak([r0, r1]) == "r1"


def test_drf_coincidence_tiebreak_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_drf_coincidence_tiebreak([])


def test_share_power_level_is_current_power_over_hardware_ceiling():
    """Unlike share_power(c) (ramp rate), this reads the candidate's own current
    instantaneous power draw -- a non-negative, purely local quantity with no coincidence/
    cancellation concern (see project discussion: aggregate power level is a sum of
    non-negative terms, so bounding each candidate's own share is a genuine, tight bound on
    the fleet aggregate, unlike ramp)."""
    c = _cand("r0", power_w=225.0, power_level_ceiling_w=450.0)
    assert share_power_level(c) == 0.5


def test_dominant_share_peak_uses_power_level_share_as_the_power_dimension():
    # compute=0.8 (dom), load=0.1, power_level=225/450=0.5
    c = _cand("r0", new_tokens=80, in_flight_after=1, token_budget=100, max_num_seqs=10,
              power_w=225.0, power_level_ceiling_w=450.0)
    assert dominant_share_peak(c) == 0.8


def test_drf_peak_power_tiebreak_routes_away_from_replica_near_its_power_ceiling():
    """Direct analog of drf_power_tiebreak's own test, using current power LEVEL instead of
    ramp rate: r0 draws almost its full hardware power limit already; r1 has full headroom.
    Primary criterion (route to lowest dominant share) must pick r1."""
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, power_w=440.0, power_level_ceiling_w=450.0)  # dom=0.978
    r1 = _cand("r1", new_tokens=50, in_flight_after=5, power_w=0.0, power_level_ceiling_w=450.0)   # dom=0.5
    assert pick_drf_peak_power_tiebreak([r0, r1]) == "r1"


def test_drf_peak_power_tiebreak_prefers_lower_power_level_over_lower_load_when_compute_ties():
    """Same tie-break-order test as drf_power_tiebreak's, using power LEVEL: two candidates
    tied on dominant compute share; r0 has more load but less power draw, r1 has less load
    but more power draw. Power-priority tie-break must pick r0 (favors low power)."""
    r0 = _cand("r0", new_tokens=90, token_budget=100, in_flight_after=7, max_num_seqs=10,
               power_w=135.0, power_level_ceiling_w=450.0)  # compute=0.9(dom), load=0.7, power=0.3
    r1 = _cand("r1", new_tokens=90, token_budget=100, in_flight_after=2, max_num_seqs=10,
               power_w=225.0, power_level_ceiling_w=450.0)  # compute=0.9(dom), load=0.2, power=0.5
    assert pick_drf([r0, r1]) == "r1"  # magnitude-sort tie-break: favors low load
    assert pick_drf_peak_power_tiebreak([r0, r1]) == "r0"  # power-priority: favors low power level


def test_drf_peak_power_tiebreak_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_drf_peak_power_tiebreak([])


def test_drf_power_tiebreak_p2c_picks_the_better_of_two_sampled_candidates():
    """Power-of-Two-Choices (Mitzenmacher, 1996/2001) applied to the DRF score itself,
    not gated to whale-only traffic like pick_p2c_whale -- every request is routed by
    sampling 2 candidates and comparing drf_power_tiebreak's own tie-break vector
    (dominant_share, power, load), same scoring as pick_drf_power_tiebreak but over a
    random 2-of-N sample instead of full visibility over every candidate."""
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=95.0, ramp_ceiling_w_per_s=100.0)   # dom=0.95
    r1 = _cand("r1", new_tokens=50, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # dom=0.5 (better)
    r2 = _cand("r2", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=99.0, ramp_ceiling_w_per_s=100.0)   # dom=0.99 (worst)
    fake_rng = _FakeRng([r0, r1])  # sample happens to exclude the worst candidate r2
    assert pick_drf_power_tiebreak_p2c([r0, r1, r2], fake_rng) == "r1"


def test_drf_power_tiebreak_p2c_never_sees_candidates_outside_its_sample():
    """If the 2-of-3 sample excludes the globally best candidate, P2C must NOT reach past
    the sample to find it -- that would defeat the whole point of bounded visibility."""
    r0 = _cand("r0", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=95.0, ramp_ceiling_w_per_s=100.0)   # dom=0.95
    r1 = _cand("r1", new_tokens=1, in_flight_after=1, ramp_rate_w_per_s=99.0, ramp_ceiling_w_per_s=100.0)   # dom=0.99 (worst)
    r2 = _cand("r2", new_tokens=50, in_flight_after=5, ramp_rate_w_per_s=0.0, ramp_ceiling_w_per_s=100.0)   # dom=0.5 (globally best)
    fake_rng = _FakeRng([r0, r1])  # sample excludes the globally-best r2
    assert pick_drf_power_tiebreak_p2c([r0, r1, r2], fake_rng) == "r0"  # best OF THE SAMPLE, not overall


def test_drf_power_tiebreak_p2c_single_candidate_fleet_needs_no_sampling():
    r0 = _cand("r0")
    assert pick_drf_power_tiebreak_p2c([r0], rng=None) == "r0"


def test_drf_power_tiebreak_p2c_raises_on_empty_candidates():
    with pytest.raises(ValueError):
        pick_drf_power_tiebreak_p2c([], rng=None)


def test_drf_power_tiebreak_p2c_with_real_rng_distributes_across_replicas():
    """Not a fake-rng unit test -- exercises the actual random.Random path end to end to
    confirm repeated routing under tied candidates doesn't collapse onto one replica."""
    import random
    cands = [_cand("r0"), _cand("r1"), _cand("r2")]  # all tied
    rng = random.Random(42)
    chosen = {pick_drf_power_tiebreak_p2c(cands, rng) for _ in range(30)}
    assert chosen == {"r0", "r1", "r2"}


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
