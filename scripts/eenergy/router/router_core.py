"""Pure orchestration: given already-tokenized request token_ids, decide which replica to
route to under one of the four policies. No network I/O, no tokenizer, no pynvml here --
those live in proxy_server.py / power_nvml.py and are called before/around this class,
keeping Router itself fully unit-testable."""
import random

from cache_mirror import new_tokens_if_routed, record_cached
from load_tracker import LoadTracker
from whale_tracker import WhaleTracker
from scoring import (Candidate, pick_round_robin, pick_lmetric, pick_drf, pick_p2c_whale,
                      pick_whale_argmin, pick_constrained_lmetric)

_POLICIES = ("round_robin", "lmetric", "drf", "p2c_whale", "whale_argmin", "constrained_lmetric")

# Admission-time whale classification cutoff (prompt tokens), reused from this project's
# existing whale-aware-controller convention (scripts/mlsys/hotpatch_whale_aware_budget.py)
# rather than a new arbitrary number.
WHALE_TOKEN_THRESHOLD = 4000


class Router:
    def __init__(self, replica_states: list, policy: str, rng=None):
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy: {policy!r}, expected one of {_POLICIES}")
        self.replica_states = replica_states
        self.policy = policy
        self.rng = rng if rng is not None else random.Random()
        self.load_tracker = LoadTracker()
        self.whale_tracker = WhaleTracker()
        self._rr_index = -1
        self._tie_cursor = 0

    def _build_candidates(self, token_ids: list) -> list:
        candidates = []
        for state in self.replica_states:
            new_tokens = new_tokens_if_routed(token_ids, state.cached_block_hashes)
            in_flight_after = self.load_tracker.in_flight_if_dispatched(state.config.replica_id)
            active_whale_count_after = self.whale_tracker.active_whale_count_if_dispatched(
                state.config.replica_id)
            candidates.append(Candidate(
                replica_id=state.config.replica_id,
                new_tokens=new_tokens,
                in_flight_after=in_flight_after,
                token_budget=state.config.token_budget,
                max_num_seqs=state.config.max_num_seqs,
                ramp_rate_w_per_s=state.ramp_rate_w_per_s,
                ramp_ceiling_w_per_s=state.config.ramp_ceiling_w_per_s,
                active_whale_count_after=active_whale_count_after,
            ))
        return candidates

    def route(self, token_ids: list) -> str:
        candidates = self._build_candidates(token_ids)
        is_whale = len(token_ids) > WHALE_TOKEN_THRESHOLD
        if self.policy == "round_robin":
            replica_id, self._rr_index = pick_round_robin(candidates, self._rr_index)
        elif self.policy == "lmetric":
            replica_id = pick_lmetric(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf":
            replica_id = pick_drf(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "p2c_whale":
            replica_id = pick_p2c_whale(candidates, is_whale, self.rng, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "whale_argmin":
            replica_id = pick_whale_argmin(candidates, is_whale, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        else:
            replica_id = pick_constrained_lmetric(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)

        self.load_tracker.on_dispatch(replica_id)
        self.whale_tracker.on_dispatch(replica_id, is_whale)
        for state in self.replica_states:
            if state.config.replica_id == replica_id:
                record_cached(token_ids, state.cached_block_hashes)
                break
        return replica_id

    def complete(self, replica_id: str, is_whale: bool = False) -> None:
        self.load_tracker.on_complete(replica_id)
        self.whale_tracker.on_complete(replica_id, is_whale)
