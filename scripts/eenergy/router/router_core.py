"""Pure orchestration: given already-tokenized request token_ids, decide which replica to
route to under one of the four policies. No network I/O, no tokenizer, no pynvml here --
those live in proxy_server.py / power_nvml.py and are called before/around this class,
keeping Router itself fully unit-testable."""
import random

from adaptive_ceiling import AdaptiveRampCeiling
from cache_mirror import new_tokens_if_routed, record_cached
from load_tracker import LoadTracker
from whale_tracker import WhaleTracker
from scoring import (Candidate, pick_round_robin, pick_lmetric, pick_drf, pick_p2c_whale,
                      pick_whale_argmin, pick_constrained_lmetric, pick_pressure_switch,
                      pick_drf_power_tiebreak, pick_lmetric_power, pick_lmetric_power_convex,
                      pick_whale_argmin_power_switch, pick_drf_coincidence_tiebreak,
                      pick_drf_peak_power_tiebreak, pick_drf_power_tiebreak_p2c)

_POLICIES = ("round_robin", "lmetric", "drf", "p2c_whale", "whale_argmin", "constrained_lmetric",
             "pressure_switch", "drf_power_tiebreak", "lmetric_power", "lmetric_power_convex",
             "whale_argmin_power_switch", "drf_coincidence_tiebreak", "drf_peak_power_tiebreak",
             "drf_power_tiebreak_p2c", "drf_power_tiebreak_adaptive")

# Admission-time whale classification cutoff (prompt tokens), reused from this project's
# existing whale-aware-controller convention (scripts/mlsys/hotpatch_whale_aware_budget.py)
# rather than a new arbitrary number.
WHALE_TOKEN_THRESHOLD = 4000


class Router:
    def __init__(self, replica_states: list, policy: str, rng=None, bs_source: str = "local",
                 whale_token_threshold: int = WHALE_TOKEN_THRESHOLD):
        if policy not in _POLICIES:
            raise ValueError(f"unknown policy: {policy!r}, expected one of {_POLICIES}")
        if bs_source not in ("local", "telemetry"):
            raise ValueError(f"unknown bs_source: {bs_source!r}, expected 'local' or 'telemetry'")
        self.replica_states = replica_states
        self.policy = policy
        self.bs_source = bs_source
        self.whale_token_threshold = whale_token_threshold
        self.rng = rng if rng is not None else random.Random()
        self.load_tracker = LoadTracker()
        self.whale_tracker = WhaleTracker()
        self.adaptive_ceiling = AdaptiveRampCeiling()
        self._rr_index = -1
        self._tie_cursor = 0
        self.last_new_tokens = None  # P-token actually used for the most recent route() call --
                                      # exposed so callers (proxy_server's assignment log) can
                                      # audit the KV$-hit discount against reality post-hoc
        self.last_is_whale = None  # is_whale decided for the most recent route() call --
                                    # exposed so callers can log/complete() consistently with
                                    # whatever whale_token_threshold this Router was built with,
                                    # instead of recomputing against a possibly different value

    def _build_candidates(self, token_ids: list) -> list:
        candidates = []
        if self.policy == "drf_power_tiebreak_adaptive":
            for state in self.replica_states:
                self.adaptive_ceiling.observe(state.ramp_rate_w_per_s)
            live_ceiling = self.adaptive_ceiling.ceiling()
        for state in self.replica_states:
            new_tokens = new_tokens_if_routed(token_ids, state.cached_block_hashes)
            if self.bs_source == "telemetry":
                in_flight_after = state.telemetry_bs + 1
            else:
                in_flight_after = self.load_tracker.in_flight_if_dispatched(state.config.replica_id)
            active_whale_count_after = self.whale_tracker.active_whale_count_if_dispatched(
                state.config.replica_id)
            if self.policy == "drf_power_tiebreak_adaptive":
                ramp_ceiling = live_ceiling
            else:
                ramp_ceiling = state.config.ramp_ceiling_w_per_s
            candidates.append(Candidate(
                replica_id=state.config.replica_id,
                new_tokens=new_tokens,
                in_flight_after=in_flight_after,
                token_budget=state.config.token_budget,
                max_num_seqs=state.config.max_num_seqs,
                ramp_rate_w_per_s=state.ramp_rate_w_per_s,
                ramp_ceiling_w_per_s=ramp_ceiling,
                active_whale_count_after=active_whale_count_after,
                power_w=state.last_power_w,
                power_level_ceiling_w=state.config.power_level_ceiling_w,
            ))
        return candidates

    def route(self, token_ids: list) -> str:
        candidates = self._build_candidates(token_ids)
        is_whale = len(token_ids) > self.whale_token_threshold
        self.last_is_whale = is_whale
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
        elif self.policy == "constrained_lmetric":
            replica_id = pick_constrained_lmetric(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak":
            replica_id = pick_drf_power_tiebreak(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power":
            replica_id = pick_lmetric_power(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_convex":
            replica_id = pick_lmetric_power_convex(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "whale_argmin_power_switch":
            replica_id = pick_whale_argmin_power_switch(candidates, is_whale, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_coincidence_tiebreak":
            replica_id = pick_drf_coincidence_tiebreak(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_peak_power_tiebreak":
            replica_id = pick_drf_peak_power_tiebreak(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_p2c":
            replica_id = pick_drf_power_tiebreak_p2c(candidates, self.rng)
        elif self.policy == "drf_power_tiebreak_adaptive":
            replica_id = pick_drf_power_tiebreak(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        else:
            replica_id = pick_pressure_switch(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)

        self.last_new_tokens = next(c.new_tokens for c in candidates if c.replica_id == replica_id)
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
