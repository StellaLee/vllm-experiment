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
