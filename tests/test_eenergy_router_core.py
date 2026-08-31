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


def test_complete_without_is_whale_stays_backward_compatible():
    r = Router(_states(), policy="round_robin")
    chosen = r.route(token_ids=[1])
    r.complete(chosen)  # no is_whale arg -- must not raise, must not touch whale_tracker
    assert r.whale_tracker.active_whale_count(chosen) == 0


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
