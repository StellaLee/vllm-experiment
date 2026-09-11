"""Pure orchestration: given already-tokenized request token_ids, decide which replica to
route to under one of the four policies. No network I/O, no tokenizer, no pynvml here --
those live in proxy_server.py / power_nvml.py and are called before/around this class,
keeping Router itself fully unit-testable."""
import random

from adaptive_ceiling import AdaptiveRampCeiling
from cache_mirror import new_tokens_if_routed, record_cached
from load_tracker import LoadTracker
from whale_tracker import WhaleTracker
from scoring import (Candidate, share_power, share_power_level, coincidence_ceiling_factor,
                      coincidence_ceiling_factor_peak,
                      pick_drf_peak_power_tiebreak_full_coincidence_ceiling,
                      pick_round_robin, pick_lmetric, pick_drf, pick_p2c_whale,
                      pick_whale_argmin, pick_constrained_lmetric, pick_pressure_switch,
                      pick_drf_power_tiebreak, pick_lmetric_power, pick_lmetric_power_convex,
                      pick_whale_argmin_power_switch, pick_drf_coincidence_tiebreak,
                      pick_drf_peak_power_tiebreak, pick_drf_power_tiebreak_p2c,
                      pick_compute_only, pick_load_only, pick_drf_power_tiebreak_full, pick_weighted_sum,
                      pick_lmetric_power_pareto, pick_drf_power_tiebreak_full_coincidence_ceiling,
                      pick_drf_no_power, pick_weighted_sum_no_power,
                      pick_weighted_sum_coincidence_ceiling, pick_lmetric_power_coincidence_ceiling,
                      pick_lmetric_power_pareto_coincidence_ceiling,
                      pick_lmetric_power_pareto_epsilon_coincidence_ceiling,
                      pick_lmetric_power_pareto_epsilon_all_coincidence_ceiling,
                      pick_lmetric_power_pareto_epsilon_small_coincidence_ceiling,
                      pick_lmetric_power_pareto_epsilon_zero_coincidence_ceiling,
                      pick_drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp,
                      pick_drf_peak_power_tiebreak_full,
                      pick_drf_power_tiebreak_full_coincidence_ceiling_random_power,
                      elevated_count, coincidence_ceiling_factor_from_count,
                      pick_drf_power_tiebreak_full_coincidence_ceiling_with_factor,
                      LAGGED_ELEVATION_DEPTH)

_POLICIES = ("round_robin", "lmetric", "drf", "p2c_whale", "whale_argmin", "constrained_lmetric",
             "pressure_switch", "drf_power_tiebreak", "lmetric_power", "lmetric_power_convex",
             "whale_argmin_power_switch", "drf_coincidence_tiebreak", "drf_peak_power_tiebreak",
             "drf_power_tiebreak_p2c", "drf_power_tiebreak_adaptive",
             "drf_power_tiebreak_adaptive_isolated", "compute_only", "drf_power_tiebreak_full",
             "weighted_sum", "lmetric_power_pareto", "drf_power_tiebreak_full_coincidence_ceiling",
             "drf_no_power", "weighted_sum_no_power", "weighted_sum_coincidence_ceiling",
             "lmetric_power_coincidence_ceiling", "load_only",
             "lmetric_power_pareto_coincidence_ceiling",
             "lmetric_power_pareto_epsilon_coincidence_ceiling",
             "lmetric_power_pareto_epsilon_all_coincidence_ceiling",
             "lmetric_power_pareto_epsilon_small_coincidence_ceiling",
             "lmetric_power_pareto_epsilon_zero_coincidence_ceiling",
             "drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp",
             "drf_peak_power_tiebreak_full",
             "drf_power_tiebreak_full_coincidence_ceiling_random_power",
             "drf_peak_power_tiebreak_full_coincidence_ceiling",
             "drf_power_tiebreak_full_coincidence_ceiling_lagged_elevation")

# Admission-time whale classification cutoff (prompt tokens), reused from this project's
# existing whale-aware-controller convention (scripts/mlsys/hotpatch_whale_aware_budget.py)
# rather than a new arbitrary number.
WHALE_TOKEN_THRESHOLD = 4000

# Safety threshold tau for the avoidable-threshold-violation diagnostic (Theorem 4/5): shares
# are normalized with 1.0 representing "at capacity" (Sec 3), so tau=1.0 is the natural
# choice -- also exactly the tau used in Theorem 5's own worked counterexample (paper.tex).
THRESHOLD_TAU = 1.0


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
        self.adaptive_ceiling_isolated = AdaptiveRampCeiling()
        self._rr_index = -1
        self._tie_cursor = 0
        self.last_new_tokens = None  # P-token actually used for the most recent route() call --
                                      # exposed so callers (proxy_server's assignment log) can
                                      # audit the KV$-hit discount against reality post-hoc
        self.last_share_compute = None  # Share_compute/Share_load for the CHOSEN candidate on
        self.last_share_load = None     # the most recent route() call -- generic (computed
                                         # regardless of policy), exposed so callers can log
                                         # which resource was dominant for the actual pick, e.g.
                                         # to diagnose whether a max()-combination rule
                                         # (drf_no_power) switches its effective dominant
                                         # resource decision-to-decision more than a
                                         # single-resource rule does
        self.last_is_whale = None  # is_whale decided for the most recent route() call --
                                    # exposed so callers can log/complete() consistently with
                                    # whatever whale_token_threshold this Router was built with,
                                    # instead of recomputing against a possibly different value
        self.last_D_chosen = None  # dominant share D(c) of the picked candidate, and the
        self.last_min_D_available = None  # minimum D(c) across the WHOLE candidate set, for
                                           # the most recent route() call -- computed
                                           # generically (policy-independent) so any rule's
                                           # picks can be audited against Theorem 4/5's
                                           # threshold-safety claim, not just the rules that
                                           # were designed to satisfy it
        self.last_avoidable_threshold_violation = None  # True iff last_D_chosen > TAU while
                                                          # last_min_D_available <= TAU: the
                                                          # rule passed over an available safe
                                                          # candidate for an unsafe one
        self._elevation_lag_buffer = []  # FIFO of real n_elevated readings, only mutated by
                                          # the lagged-elevation-count ablation policy
        self._last_lagged_cc_factor = 1.0  # the ceiling-scaling factor actually used by the
                                            # most recent lagged-elevation-count route() call --
                                            # reused (not recomputed) by the threshold-safety
                                            # diagnostic below so it doesn't double-consume
                                            # _elevation_lag_buffer

    def _build_candidates(self, token_ids: list) -> list:
        candidates = []
        if self.policy == "drf_power_tiebreak_adaptive":
            for state in self.replica_states:
                self.adaptive_ceiling.observe(state.ramp_rate_w_per_s)
            live_ceiling = self.adaptive_ceiling.ceiling()
        elif self.policy == "drf_power_tiebreak_adaptive_isolated":
            self.adaptive_ceiling_isolated.observe_round(
                [state.ramp_rate_w_per_s for state in self.replica_states])
            live_ceiling = self.adaptive_ceiling_isolated.ceiling()
        for state in self.replica_states:
            new_tokens = new_tokens_if_routed(token_ids, state.cached_block_hashes)
            if self.bs_source == "telemetry":
                in_flight_after = state.telemetry_bs + 1
            else:
                in_flight_after = self.load_tracker.in_flight_if_dispatched(state.config.replica_id)
            active_whale_count_after = self.whale_tracker.active_whale_count_if_dispatched(
                state.config.replica_id)
            if self.policy in ("drf_power_tiebreak_adaptive", "drf_power_tiebreak_adaptive_isolated"):
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
                smoothed_ramp_rate_w_per_s=state.smoothed_ramp_rate_w_per_s,
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
        elif self.policy == "drf_power_tiebreak_adaptive_isolated":
            replica_id = pick_drf_power_tiebreak(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "compute_only":
            replica_id = pick_compute_only(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "load_only":
            replica_id = pick_load_only(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_full":
            replica_id = pick_drf_power_tiebreak_full(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "weighted_sum":
            replica_id = pick_weighted_sum(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto":
            replica_id = pick_lmetric_power_pareto(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_full_coincidence_ceiling":
            replica_id = pick_drf_power_tiebreak_full_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_no_power":
            replica_id = pick_drf_no_power(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "weighted_sum_no_power":
            replica_id = pick_weighted_sum_no_power(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "weighted_sum_coincidence_ceiling":
            replica_id = pick_weighted_sum_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_coincidence_ceiling":
            replica_id = pick_lmetric_power_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto_coincidence_ceiling":
            replica_id = pick_lmetric_power_pareto_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto_epsilon_coincidence_ceiling":
            replica_id = pick_lmetric_power_pareto_epsilon_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto_epsilon_all_coincidence_ceiling":
            replica_id = pick_lmetric_power_pareto_epsilon_all_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto_epsilon_small_coincidence_ceiling":
            replica_id = pick_lmetric_power_pareto_epsilon_small_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "lmetric_power_pareto_epsilon_zero_coincidence_ceiling":
            replica_id = pick_lmetric_power_pareto_epsilon_zero_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp":
            replica_id = pick_drf_power_tiebreak_full_coincidence_ceiling_smoothed_ramp(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_peak_power_tiebreak_full":
            replica_id = pick_drf_peak_power_tiebreak_full(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_full_coincidence_ceiling_random_power":
            replica_id = pick_drf_power_tiebreak_full_coincidence_ceiling_random_power(
                candidates, self.rng, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_peak_power_tiebreak_full_coincidence_ceiling":
            replica_id = pick_drf_peak_power_tiebreak_full_coincidence_ceiling(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        elif self.policy == "drf_power_tiebreak_full_coincidence_ceiling_lagged_elevation":
            real_n_elevated = elevated_count(candidates)
            self._elevation_lag_buffer.append(real_n_elevated)
            if len(self._elevation_lag_buffer) > LAGGED_ELEVATION_DEPTH:
                lagged_n_elevated = self._elevation_lag_buffer.pop(0)
            else:
                lagged_n_elevated = real_n_elevated
            self._last_lagged_cc_factor = coincidence_ceiling_factor_from_count(lagged_n_elevated)
            replica_id = pick_drf_power_tiebreak_full_coincidence_ceiling_with_factor(
                candidates, self._last_lagged_cc_factor, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)
        else:
            replica_id = pick_pressure_switch(candidates, self._tie_cursor)
            self._tie_cursor = (self._tie_cursor + 1) % len(candidates)

        self.last_new_tokens = next(c.new_tokens for c in candidates if c.replica_id == replica_id)
        chosen = next(c for c in candidates if c.replica_id == replica_id)
        self.last_share_compute = chosen.new_tokens / chosen.token_budget
        self.last_share_load = chosen.in_flight_after / chosen.max_num_seqs

        # Threshold-safety diagnostic (Theorem 4/5): whatever ceiling THIS policy actually
        # uses -- the coincidence-ceiling-adjusted one for any "_coincidence_ceiling" variant,
        # the raw static ceiling otherwise -- computed generically so every policy's picks can
        # be audited against the same claim, not just the ones designed to satisfy it.
        is_peak_variant = self.policy == "drf_peak_power_tiebreak_full_coincidence_ceiling"
        is_lagged_variant = self.policy == "drf_power_tiebreak_full_coincidence_ceiling_lagged_elevation"
        if is_peak_variant:
            cc_factor = coincidence_ceiling_factor_peak(candidates)
        elif is_lagged_variant:
            # Reuse the factor actually used above -- recomputing here would read (and
            # mutate) _elevation_lag_buffer a second time for the same decision.
            cc_factor = self._last_lagged_cc_factor
        elif "coincidence_ceiling" in self.policy:
            cc_factor = coincidence_ceiling_factor(candidates)
        else:
            cc_factor = 1.0

        def _dominant_share(c):
            sc = c.new_tokens / c.token_budget
            sl = c.in_flight_after / c.max_num_seqs
            sp = (share_power_level(c) if is_peak_variant else share_power(c)) / cc_factor
            return max(sc, sl, sp)

        all_D = [_dominant_share(c) for c in candidates]
        self.last_D_chosen = _dominant_share(chosen)
        self.last_min_D_available = min(all_D)
        self.last_avoidable_threshold_violation = (
            self.last_D_chosen > THRESHOLD_TAU and self.last_min_D_available <= THRESHOLD_TAU)

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
