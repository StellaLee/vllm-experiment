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


def dominant_share(c) -> float:
    """Share_compute, Share_load, Share_power for one candidate, each normalized to that
    replica's own configured capacity (no cross-resource weight); returns the max, i.e. the
    dominant share (spec S3.1). A negative ramp rate (power decreasing) never counts as
    pressure."""
    share_compute = c.new_tokens / c.token_budget
    share_load = c.in_flight_after / c.max_num_seqs
    share_power = max(c.ramp_rate_w_per_s, 0.0) / c.ramp_ceiling_w_per_s
    return max(share_compute, share_load, share_power)


def pick_drf(candidates: list, tie_start: int = 0) -> str:
    """Condition 3 (ours). Route to the replica with the lowest resulting dominant share
    across the three independently-normalized resources -- Dominant Resource Fairness
    (Ghodsi et al., NSDI 2011), adapted to an online per-request routing setting (spec
    S3.1, with the honest scope caveat that DRF's original theorem is proven for a static
    allocation game, not this streaming setting). tie_start rotates which candidate wins
    ties (see _rotate) -- Router advances it every call so ties distribute across replicas
    over time instead of piling onto one replica."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(_rotate(candidates, tie_start), key=dominant_share)
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
