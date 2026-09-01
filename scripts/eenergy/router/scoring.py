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


def lmetric_power_score(c) -> float:
    """LMETRIC's own multiplicative form (Zhang et al., OSDI'26, arXiv:2603.15202),
    extended with a continuous power penalty: new_tokens x in_flight_after x
    (1 + share_power). Route to the minimum -- no separate tie-break rule needed at all,
    unlike DRF's max()-of-shares family (dominant_share / dominant_share_vector /
    dominant_share_vector_power_priority), where whichever dimension is currently largest
    silences the other two entirely (the root cause diagnosed this session for every DRF
    tie-break failure mode: compute silencing power for whales, load silencing power by
    accident of magnitude, power silencing load under light load once "fixed" to check it
    first). A product never does that -- every factor always contributes something,
    proportionally, regardless of which is numerically biggest. (1 + share_power) is bounded
    and always positive: a replica at its ramp ceiling (share_power=1) scores 2x worse than
    an otherwise-identical replica with zero ramp; one with real headroom is barely
    penalized. A lightweight, always-on, continuous cousin of Neely's Lyapunov
    drift-plus-penalty (Stochastic Network Optimization, 2010) -- already cited here for
    constrained_lmetric's hard V-to-infinity feasibility cutoff -- but smooth instead of a
    discrete on/off boundary, so there's no threshold to misfire near."""
    return c.new_tokens * c.in_flight_after * (1.0 + share_power(c))


def pick_lmetric_power(candidates: list, tie_start: int = 0) -> str:
    """Route to the candidate with the minimum lmetric_power_score. tie_start rotates which
    candidate wins residual exact ties (see _rotate), matching every other picker's
    convention."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=lmetric_power_score)
    return best.replica_id


def lmetric_power_convex_score(c) -> float:
    """Same LMETRIC-style multiplicative form as lmetric_power_score, but a CONVEX power
    penalty -- new_tokens x in_flight_after x (1 + share_power^2) -- instead of linear.
    Targets the mean_ramp/duty_cycle regression diagnosed for lmetric_power under real load
    (findings.md, 2026-09-01): a linear penalty reacts to noise-level power differences even
    when every replica is comfortably under-ceiling, plausibly causing a chase-the-coolest-
    replica oscillation that raised mean ramp activity. Squaring makes the penalty much
    smaller than linear below the ceiling (0.3^2=0.09 vs 0.3 -- barely reactive to noise)
    while still growing sharply near/above it (matches linear exactly at share_power=1,
    overtakes it beyond). Convex ramp-cost penalties are the standard convention in
    power-systems economic dispatch / unit commitment literature: stressing a generator near
    its ramp limit carries disproportionate, super-linear cost -- arguably better-grounded
    for this project's actual domain than the linear version, not just an ad hoc tweak."""
    sp = share_power(c)
    return c.new_tokens * c.in_flight_after * (1.0 + sp * sp)


def pick_lmetric_power_convex(candidates: list, tie_start: int = 0) -> str:
    """Route to the candidate with the minimum lmetric_power_convex_score. tie_start rotates
    which candidate wins residual exact ties (see _rotate), matching every other picker's
    convention."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=lmetric_power_convex_score)
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


def dominant_share_vector_power_priority(c) -> tuple:
    """Tie-break vector for pick_drf_power_tiebreak: (dominant_share, share_power,
    share_load) -- a fixed resource PRIORITY order, not dominant_share_vector's magnitude
    sort. dominant_share_vector sorts the three shares purely by numeric size, so whichever
    of {load, power} happens to be numerically larger at that moment wins the tie-break --
    an accident of scale, not a deliberate choice that power should matter more for this
    project's ramp-protection goal. This variant always compares power second, regardless of
    its size relative to load, so a tied dominant share breaks toward the replica with less
    power pressure, not just less of whichever of {load, power} happens to be bigger."""
    share_load = c.in_flight_after / c.max_num_seqs
    sp = share_power(c)
    return (dominant_share(c), sp, share_load)


def pick_drf_power_tiebreak(candidates: list, tie_start: int = 0) -> str:
    """DRF variant: same primary criterion as pick_drf (route to the replica with the
    lowest dominant share), but breaks ties using dominant_share_vector_power_priority's
    fixed (power, then load) resource order instead of dominant_share_vector's magnitude
    sort. Targets the diagnosed mechanism (findings.md, 2026-09-01) where Share_power gets
    accidentally outranked by Share_load in ties purely because load happens to be
    numerically larger at that moment -- not because power is less relevant to what this
    project is actually trying to protect."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=dominant_share_vector_power_priority)
    return best.replica_id


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


def fleet_pressured(candidates: list) -> bool:
    """True iff ANY replica in the fleet currently exceeds its own calibrated ramp ceiling
    (share_power > 1.0) -- a global, directly-observable signal already computed for every
    candidate on every decision, no new telemetry needed. Feeds pick_pressure_switch's mode
    select; not itself a routing decision."""
    return any(share_power(c) > 1.0 for c in candidates)


def pick_pressure_switch(candidates: list, tie_start: int = 0) -> str:
    """Fleet-state-triggered policy switching, not a new scoring formula: route via DRF
    (lexicographic tie-break, see dominant_share_vector) whenever the fleet is CURRENTLY in a
    power-pressure window (fleet_pressured), else route via plain LMETRIC. Mirrors the
    sibling mlsys project's whale-aware budget controller (token_budget = 512 if
    whale_active else 16384) -- switch between two independently-validated mechanisms on
    directly-observed global state rather than trying to make one scoring formula do both
    jobs at once (the failure mode of p2c_whale, whale_argmin, and constrained_lmetric).

    Deliberately fleet-wide, not per-candidate: a single pressured replica anywhere changes
    the policy for THIS decision, even if the request wouldn't have landed on that replica
    under LMETRIC anyway -- the point is to avoid growing the pressure window while it's
    open, not merely to dodge the one hot replica."""
    if not candidates:
        raise ValueError("no candidates to route to")
    if fleet_pressured(candidates):
        return pick_drf(candidates, tie_start)
    return pick_lmetric(candidates, tie_start)


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


def pick_whale_argmin_power_switch(candidates: list, is_whale: bool, tie_start: int = 0) -> str:
    """Fleet-state-triggered switching, same pattern as pick_pressure_switch, but with
    whale_argmin as the default instead of plain LMETRIC: route via drf_power_tiebreak
    whenever the fleet is CURRENTLY power-pressured (fleet_pressured), else via whale_argmin
    (which itself already delegates non-whale traffic to LMETRIC unchanged).

    Motivation (findings.md, 2026-09-01): whale_argmin won TTFT outright in every
    whale-injection-based condition tested this session (closed-loop, both open-loop rates,
    both decode-cap settings, throughput-matched) -- a stronger, more consistently-validated
    default than plain LMETRIC. Its one real, bounded weakness is TBT-mean/mean-ramp under
    real load -- exactly what drf_power_tiebreak (the resource-priority DRF tie-break fix,
    validated specifically under real ramp pressure) is good at. Gating the switch to only
    engage during genuine pressure should inherit whale_argmin's proven TTFT record
    everywhere it already wins, while only paying drf_power_tiebreak's cost during the
    windows its ramp protection is actually needed -- and should avoid drf_power_tiebreak's
    own documented light-load regression, since it would rarely engage there by construction
    (no pressure = no engagement). Not yet validated; this is the constructed hypothesis, not
    a result."""
    if not candidates:
        raise ValueError("no candidates to route to")
    if fleet_pressured(candidates):
        return pick_drf_power_tiebreak(candidates, tie_start)
    return pick_whale_argmin(candidates, is_whale, tie_start)
