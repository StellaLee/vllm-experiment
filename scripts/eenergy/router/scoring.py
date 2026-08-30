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


def pick_round_robin(candidates: list, last_index: int):
    """Condition 1 (vLLM default -- vLLM ships no built-in multi-replica router, so
    round-robin is the honest content-/state-blind baseline; see spec S3.3). Ignores every
    per-candidate metric; cycles through the given candidate order. Returns (chosen
    replica_id, new last_index to store for the next call)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    next_index = (last_index + 1) % len(candidates)
    return candidates[next_index].replica_id, next_index


def pick_lmetric(candidates: list) -> str:
    """Condition 2. Score = P-token x BS, route to the minimum (Zhang et al., "Simple is
    Better: Multiplication May Be All You Need for LLM Request Scheduling", OSDI'26,
    arXiv:2603.15202)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(candidates, key=lambda c: c.new_tokens * c.in_flight_after)
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


def pick_drf(candidates: list) -> str:
    """Condition 3 (ours). Route to the replica with the lowest resulting dominant share
    across the three independently-normalized resources -- Dominant Resource Fairness
    (Ghodsi et al., NSDI 2011), adapted to an online per-request routing setting (spec
    S3.1, with the honest scope caveat that DRF's original theorem is proven for a static
    allocation game, not this streaming setting)."""
    if not candidates:
        raise ValueError("no candidates to route to")
    best = min(candidates, key=dominant_share)
    return best.replica_id
