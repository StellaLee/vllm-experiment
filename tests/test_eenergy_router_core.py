import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from router_core import Router  # noqa: E402
from replica_state import ReplicaConfig, ReplicaState  # noqa: E402


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
