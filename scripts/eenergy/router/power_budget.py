"""Rolling-window fleet-power budget for the peak-shaving admission gate (see
docs/superpowers/specs/2026-09-11-peak-shaving-admission-design.md). Pure, hardware-free --
power_poll_loop (proxy_server.py) feeds real samples in; the admission gate in
handle_completions reads it before routing."""
from collections import deque


class PowerBudget:
    """Tracks a rolling window of fleet-aggregate power samples and answers whether admitting
    a request with a given marginal energy cost would push the trailing window's average
    power over a cap. Fails OPEN (would_exceed returns False) when the window has fewer than
    2 samples -- at startup, before power_poll_loop has produced enough history, blocking all
    traffic would be worse than the brief risk of an unenforced cap during warmup."""

    def __init__(self, window_s: float):
        self.window_s = window_s
        self._samples = deque()  # (t, fleet_power_w), oldest first
        self._reserved_j = 0.0  # sum of marginal_j for admitted-but-not-yet-completed requests

    def record(self, t: float, fleet_power_w: float) -> None:
        self._samples.append((t, fleet_power_w))
        cutoff = t - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def reserve(self, marginal_j: float) -> None:
        """Call the instant a request is admitted (right after would_exceed returns False),
        BEFORE any real power measurement could possibly reflect its draw. Closes a real
        time-of-check-to-time-of-use gap: without this, several requests arriving within the
        same power_poll_loop interval each check against the same stale real-power reading and
        can all pass, collectively overshooting the cap before the real trace catches up."""
        self._reserved_j += marginal_j

    def release(self, marginal_j: float) -> None:
        """Call when a previously-reserved request completes (handle_completions's existing
        finally block) -- by then its real draw has already accumulated into the ongoing power
        samples over its actual lifetime, so the reservation has done its job. Floored at 0 so
        a mismatched/late release can't push the ledger negative and let an oversized request
        through unrealistically."""
        self._reserved_j = max(0.0, self._reserved_j - marginal_j)

    def would_exceed(self, cap_w: float, marginal_j: float) -> bool:
        if len(self._samples) < 2:
            return False
        mean_power_w = sum(p for _, p in self._samples) / len(self._samples)
        # Account for this candidate's own marginal energy AND every currently-outstanding
        # reservation from other admitted-but-not-yet-reflected requests -- not just the
        # stale real-power reading alone.
        projected_avg_w = mean_power_w + (self._reserved_j + marginal_j) / self.window_s
        return projected_avg_w > cap_w


def estimate_marginal_energy_j(prompt_tokens: int, expected_decode_tokens: float,
                                j_per_prefill_token: float, j_per_decode_token: float) -> float:
    """Two-term marginal energy estimate for an admission-gate check, made BEFORE routing.
    prompt_tokens is the raw, un-cache-discounted prompt length (known immediately from the
    tokenizer, before routing) -- see docs/superpowers/specs/2026-09-11-peak-shaving-admission-
    design.md Sec 4.2 for why calling Router.route() speculatively isn't safe here (it has real
    dispatch-tracking side effects). expected_decode_tokens comes from DecodeByteEstimator
    below, not from a live tokenizer call on the (not-yet-generated) response.

    j_per_prefill_token/j_per_decode_token are calibrated from a dedicated isolation-burst
    hardware experiment (orchestrate/eenergy/run_prefill_decode_calibration.sh), not a
    trial-level regression (which failed -- see the spec). j_per_decode_token is robust
    (CV~3-8% across 3 replicated trials); j_per_prefill_token is not (CV~61-77%, likely
    GPU-clock-ramp-confounded) and is deliberately set to the highest observed rate rather
    than the mean -- a conservative upper bound, not a placeholder for a future fix. See the
    spec for why this is an acceptable tradeoff: decode costs 37-115x more per token than
    prefill in every calibration trial, so prefill's imprecision barely moves the estimate for
    typical short-prompt requests and only meaningfully affects the whale-heavy minority --
    where erring high is exactly the safe direction."""
    return prompt_tokens * j_per_prefill_token + expected_decode_tokens * j_per_decode_token


class DecodeByteEstimator:
    """Live EMA of realized response byte-length, updated as requests complete. Tracks BYTES,
    not tokens, so handle_completions can feed it directly from the byte counts it already
    sees while streaming a response through -- no mid-stream tokenizer call needed. Converts
    to a token-count estimate via a caller-supplied bytes-per-token constant at the point of
    use (current_estimate_tokens), keeping the EMA itself unit-agnostic."""

    def __init__(self, smoothing_alpha: float = 0.3):
        self.smoothing_alpha = smoothing_alpha
        self._estimate_bytes = None  # None until warm (no completion observed yet)

    def record_completion(self, response_bytes: int) -> None:
        if self._estimate_bytes is None:
            self._estimate_bytes = float(response_bytes)
        else:
            self._estimate_bytes = (self.smoothing_alpha * response_bytes
                                     + (1.0 - self.smoothing_alpha) * self._estimate_bytes)

    def current_estimate_tokens(self, fallback_tokens: float, bytes_per_token: float) -> float:
        """Live estimate converted to tokens, or fallback_tokens if no completion has been
        observed yet (cold start). The caller passes the CURRENT request's own max_tokens as
        the fallback -- a safe-direction default for the brief warmup window only; once warm,
        the live EMA takes over and is no longer max_tokens-biased (see the spec for why using
        max_tokens as a steady-state estimate, not just a cold-start fallback, was rejected --
        it overestimates typical output length by ~3x on this design's target conditions)."""
        if self._estimate_bytes is None:
            return fallback_tokens
        return self._estimate_bytes / bytes_per_token


def dual_budget_would_exceed(budget_prefill, cap_prefill_w: float, prefill_marginal_j: float,
                              budget_decode, cap_decode_w: float,
                              decode_marginal_j: float) -> bool:
    """True if EITHER pool's budget would be exceeded by admitting this request -- used by
    the disaggregated gate (disagg_gate.py), which tracks the prefill pool and decode pool as
    two independent PowerBudget instances instead of one combined fleet budget (spec
    docs/superpowers/specs/2026-09-11-disagg-peak-shaving-gate-design.md Sec 4.1). A None
    budget means that pool's cap is disabled -- matches proxy_server.py's existing
    single-budget convention (budget=None -> gate never blocks) -- and never contributes to
    the OR, regardless of the marginal_j passed for it."""
    prefill_exceeds = (budget_prefill is not None
                        and budget_prefill.would_exceed(cap_prefill_w, prefill_marginal_j))
    decode_exceeds = (budget_decode is not None
                       and budget_decode.would_exceed(cap_decode_w, decode_marginal_j))
    return prefill_exceeds or decode_exceeds
