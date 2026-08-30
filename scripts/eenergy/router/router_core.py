"""Pure orchestration: given already-tokenized request token_ids, decide which replica to
route to under one of the three policies. No network I/O, no tokenizer, no pynvml here --
those live in proxy_server.py / power_nvml.py and are called before/around this class,
keeping Router itself fully unit-testable."""
from cache_mirror import new_tokens_if_routed, record_cached
from load_tracker import LoadTracker
from scoring import Candidate, pick_round_robin, pick_lmetric, pick_drf

_POLICIES = ("round_robin", "lmetric", "drf")


class Router:
    def __init__(self, replica_states: list, policy: str):
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy: {policy!r}, expected one of {_POLICIES}")
        self.replica_states = replica_states
        self.policy = policy
        self.load_tracker = LoadTracker()
        self._rr_index = -1

    def _build_candidates(self, token_ids: list) -> list:
        candidates = []
        for state in self.replica_states:
            new_tokens = new_tokens_if_routed(token_ids, state.cached_block_hashes)
            in_flight_after = self.load_tracker.in_flight_if_dispatched(state.config.replica_id)
            candidates.append(Candidate(
                replica_id=state.config.replica_id,
                new_tokens=new_tokens,
                in_flight_after=in_flight_after,
                token_budget=state.config.token_budget,
                max_num_seqs=state.config.max_num_seqs,
                ramp_rate_w_per_s=state.ramp_rate_w_per_s,
                ramp_ceiling_w_per_s=state.config.ramp_ceiling_w_per_s,
            ))
        return candidates

    def route(self, token_ids: list) -> str:
        candidates = self._build_candidates(token_ids)
        if self.policy == "round_robin":
            replica_id, self._rr_index = pick_round_robin(candidates, self._rr_index)
        elif self.policy == "lmetric":
            replica_id = pick_lmetric(candidates)
        else:
            replica_id = pick_drf(candidates)

        self.load_tracker.on_dispatch(replica_id)
        for state in self.replica_states:
            if state.config.replica_id == replica_id:
                record_cached(token_ids, state.cached_block_hashes)
                break
        return replica_id

    def complete(self, replica_id: str) -> None:
        self.load_tracker.on_complete(replica_id)
