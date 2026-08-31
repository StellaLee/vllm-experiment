"""Pure routing-decision functions for the three evaluation conditions (spec S3.3). Each
takes already-computed per-candidate metrics and returns a routing decision -- no I/O, no
network, no tokenizer -- so these are the most heavily unit-tested module in the package;
this is the actual intellectual contribution being evaluated."""
from dataclasses import dataclass


@dataclass
class Candidate:
    replica_id: str
    new_tokens: int           # P-token: new prefill tokens needed if routed here
    in_flight_after: int      # BS: in-flight count if this request were dispatched here
    token_budget: int
    max_num_seqs: int
    ramp_rate_w_per_s: float
    ramp_ceiling_w_per_s: float
    active_whale_count_after: int = 0  # active whales at this replica if dispatched here


def pick_round_robin(candidates: list, last_index: int):
    """Condition 1 (vLLM default -- vLLM ships no built-in multi-replica router, so
    round-robin is the honest content-/state-blind baseline; see spec S3.3). Ignores every
    per-candidate metric; cycles through the given candidate order. Returns (chosen
    replica_id, new last_index to store for the next call)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    next_index = (last_index + 1) % len(candidates)
    return candidates[next_index].replica_id, next_index


def _rotate(candidates: list, start: int) -> list:
    """Rotate candidates so tie-breaking (Python's min() keeps the first minimal element)
    doesn't always favor the same replica -- under a workload where a resource dimension is
    near-identical across fresh candidates (e.g. an un-cacheable prefill making every
    replica's P-token equal), ties are constant and an unrotated min() piles every tied
    request onto candidates[0], collapsing load balance across the whole fleet."""
    n = len(candidates)
    start = start % n
    return candidates[start:] + candidates[:start]


def pick_lmetric(candidates: list, tie_start: int = 0) -> str:
    """Condition 2. Score = P-token x BS, route to the minimum (Zhang et al., "Simple is
    Better: Multiplication May Be All You Need for LLM Request Scheduling", OSDI'26,
    arXiv:2603.15202). tie_start rotates which candidate wins ties (see _rotate) --
    Router advances it every call so ties distribute across replicas over time."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=lambda c: c.new_tokens * c.in_flight_after)
    return best.replica_id


def share_power(c) -> float:
    """Fraction of a replica's calibrated ramp ceiling currently in use. A negative ramp
    rate (power decreasing) never counts as pressure -- a routing decision can only ever
    push the RECEIVING replica's power up, never down, so only positive ramp is a
    decision-relevant hazard. Shared by dominant_share (DRF) and pick_constrained_lmetric
    (the hard-constraint filter) so both read the exact same power signal."""
    return max(c.ramp_rate_w_per_s, 0.0) / c.ramp_ceiling_w_per_s


def dominant_share(c) -> float:
    """Share_compute, Share_load, Share_power for one candidate, each normalized to that
    replica's own configured capacity (no cross-resource weight); returns the max, i.e. the
    dominant share (spec S3.1)."""
    share_compute = c.new_tokens / c.token_budget
    share_load = c.in_flight_after / c.max_num_seqs
    return max(share_compute, share_load, share_power(c))


def dominant_share_vector(c) -> tuple:
    """Full (compute, load, power) share vector, sorted descending. dominant_share(c) is
    always vec[0] -- this is a strict refinement for tie-breaking, not a different metric.

    Real DRF (Ghodsi et al., NSDI 2011; Mesos's own reference implementation) breaks ties
    by comparing the full sorted share vector lexicographically -- first the dominant
    share, then the next-highest, and so on -- not by falling back to an arbitrary
    rotation. Using only the scalar max (as an earlier version of pick_drf did) is a real
    deviation from the textbook mechanism: whenever one dimension dominates identically
    across every candidate (e.g. an early, un-cacheable whale makes Share_compute tied and
    dominant for everyone), the scalar max can't see differences in the OTHER dimensions
    at all, discarding exactly the information (e.g. load) that should decide the tie."""
    share_compute = c.new_tokens / c.token_budget
    share_load = c.in_flight_after / c.max_num_seqs
    return tuple(sorted((share_compute, share_load, share_power(c)), reverse=True))


def pick_drf(candidates: list, tie_start: int = 0) -> str:
    """Condition 3 (ours). Route to the replica with the lexicographically lowest sorted
    share vector across the three independently-normalized resources -- Dominant Resource
    Fairness (Ghodsi et al., NSDI 2011), adapted to an online per-request routing setting
    (spec S3.1, with the honest scope caveat that DRF's original theorem is proven for a
    static allocation game, not this streaming setting). tie_start rotates which candidate
    wins the (now rare) case of a FULL tie across all three shares (see _rotate) -- Router
    advances it every call so those residual ties distribute across replicas over time
    instead of piling onto one replica."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=dominant_share_vector)
    return best.replica_id


def pick_p2c_whale(candidates: list, is_whale: bool, rng, tie_start: int = 0) -> str:
    """Whale-only Power of Two Choices (Mitzenmacher, "The Power of Two Choices in
    Randomized Load Balancing", 1996/2001): sampling 2 random candidates and picking the
    less-loaded one gives an exponential improvement in expected max load over random
    placement, with no cross-resource weight and no per-decision power telemetry needed --
    load here is active whale count, known from admission-time request length alone.

    Non-whale requests (the overwhelming majority of traffic) delegate straight to
    pick_lmetric, unchanged -- this policy only touches routing for whale-classified
    requests, so normal-traffic TTFT/TBT/utilization should be identical to the LMETRIC
    condition. Whale requests ignore LMETRIC's score entirely and compare only the 2 sampled
    candidates' active_whale_count_after -- deliberately O(1) work per decision, not an
    argmin over the whole fleet (that would be a different, non-P2C mechanism)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    if not is_whale:
        return pick_lmetric(candidates, tie_start)
    if len(candidates) == 1:
        return candidates[0].replica_id
    sampled = rng.sample(candidates, 2)
    best = min(sampled, key=lambda c: c.active_whale_count_after)
    return best.replica_id


def pick_constrained_lmetric(candidates: list, tie_start: int = 0) -> str:
    """Constrained-LMETRIC: optimize LMETRIC's score subject to a hard feasibility
    constraint (share_power <= 1.0, i.e. not currently exceeding the calibrated ramp
    ceiling) -- the V-to-infinity limit of Lyapunov drift-plus-penalty (Neely, *Stochastic
    Network Optimization*, 2010): that framework's [O(1/V), O(V)] objective-vs-violation
    tradeoff curve has one endpoint that needs no tunable weight, because there's nothing
    left to tune once V is fixed at the limit. Reduces to plain LMETRIC exactly whenever no
    replica is over its ramp ceiling (the common case, since the ceiling rarely binds) --
    unlike DRF's dominant_share, which lets power compete with compute/load on every
    decision even at a marginal, non-dangerous share.

    If EVERY candidate is simultaneously infeasible (no replica currently safe), falls back
    to whichever is LEAST infeasible (lowest share_power) rather than silently reverting to
    plain LMETRIC -- that fallback would defeat the constraint in exactly the moment it's
    supposed to matter most."""
    if not candidates:
        raise ValueError("no candidates to route to")
    feasible = [c for c in candidates if share_power(c) <= 1.0]
    if feasible:
        return pick_lmetric(feasible, tie_start)
    best = min(_rotate(candidates, tie_start), key=share_power)
    return best.replica_id


def pick_whale_argmin(candidates: list, is_whale: bool, tie_start: int = 0) -> str:
    """Ablation for pick_p2c_whale: isolates whether P2C's 2-of-N sampling is the reason it
    doesn't beat DRF's coincidence-frequency figure (spec: DRF sees full 6-candidate ramp
    state on every decision; P2C only samples 2). Same whale-only design as pick_p2c_whale
    (non-whale traffic delegates to pick_lmetric, unchanged), but whale requests compare
    EVERY candidate's active_whale_count_after -- full visibility, no sampling, no
    randomness needed. If this closes the coincidence-frequency gap to DRF, the gap was a
    sampling-size artifact, not a fundamental property of anticipatory whale-count routing;
    if it doesn't, the gap is more fundamental than sample size."""
    if not candidates:
        raise ValueError("no candidates to route to")
    if not is_whale:
        return pick_lmetric(candidates, tie_start)
    best = min(_rotate(candidates, tie_start), key=lambda c: c.active_whale_count_after)
    return best.replica_id
