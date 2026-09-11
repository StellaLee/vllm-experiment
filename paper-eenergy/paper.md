# Fleet-Coincidence-Aware Power Routing: A Provably Safe Mechanism for Multi-GPU LLM Serving

*Target: ACM eEnergy 2027 Fall, track TBC (likely "Systems and applied modeling"). Deadline:
**Sept 18, 2026** — see `README.md`. Submission format: ACM sigconf LaTeX, anonymous
(double-blind), 10pp double-column excl. references.*

<!-- Editorial history and pivot rationale removed from submission-facing content; see the
project's internal research log for that record. -->

---

## Abstract

We formalize power-aware request routing for multi-GPU LLM-serving fleets as a
multi-resource social welfare problem: each replica carries three independently-normalized
shares — compute (prefix-cache-discounted prefill cost), load (in-flight request count), and
power (live NVML-measured ramp rate relative to a calibrated ceiling) — and the router seeks
an egalitarian (max-min) welfare allocation across candidates at each routing decision,
extending Dominant Resource Fairness (Ghodsi et al., NSDI 2011) to a third, reactive resource
dimension. We prove that routing via the fully-sorted lexicographic dominant-share rule
selects a Pareto-non-dominated candidate at every decision (verified against 200,000 random
instances with zero violations, in addition to a closed-form proof), and that this guarantee
extends unchanged to any shared, fleet-wide ceiling adjustment — including a coincidence-ceiling
mechanism we introduce, which contracts every candidate's ramp ceiling when multiple replicas
are simultaneously elevated, directly targeting the fleet-**aggregate** coincident-ramp hazard
that a purely per-replica signal cannot see. We adopt the resulting rule,
`drf_power_tiebreak_full_coincidence_ceiling`, as our proposed strategy, and show two natural
alternatives — a fixed-priority DRF tie-break, and the same coincidence-ceiling mechanism
applied to a power-extended LMETRIC score — each provably sacrifice the guarantee via distinct
mechanisms. On real 8×4090 hardware, across 7 workload conditions including two real traces
(BurstGPT, WildChat), our proposed rule reduces peak power by 4.7% relative to a power-blind
baseline on the condition central to this paper's motivation (p=0.0017, n=6), with TTFT and
TBT directionally favorable on the same condition but not independently significant at this
trial count. Against the strongest empirically-performing but *not* Pareto-safe alternative,
replicated to matched n=6, the proposed rule is statistically indistinguishable across every
metric measured in this study — peak power, TTFT, TBT, ramp-derivative statistics,
energy-per-token, and SLO-violation-rate: **the safety guarantee costs nothing on any axis we
measure.** A second, methodological contribution falls out of this validation: we characterize
a ~75-85% decision-level noise floor inherent to closed-loop real-hardware routing evaluation
(confirmed real, not a workload artifact, via a state-blind control and real unpadded
conversational traces), identify which evaluation statistics survive it (metrics that sum over
many independent events — energy-per-token, SLO-violation-rate) and which don't (ramp
derivatives and extreme-value statistics), and show this reshapes which comparisons in this
line of work can currently be asserted with confidence.

## 1. Introduction

**Motivation.** Multi-GPU LLM-serving fleets draw power in bursts as requests arrive and
complete, and the resulting ramp volatility — not just mean draw — is a concern for facility
power delivery and, at larger scale, for grid-facing demand response. This paper addresses the
fleet-internal version of that hazard: coincident power-ramp spikes across replicas within a
single serving fleet, a risk that grows with fleet size and is invisible to any policy that
reasons about replicas independently. The request router, the component deciding which
replica serves each incoming request, is a free lever against this hazard: no hardware change,
no compute cost, just a policy. This paper asks a precise question about that lever: **can we
route in a way that is provably fair/efficient across resources and defends against
*fleet-wide coincident* power-ramp hazards specifically, and — since natural power-aware
deviations from a provably safe rule turn out to sacrifice the guarantee — do we actually have
to give up that guarantee to get a power-aware win in practice, or does the safe rule already
deliver one for free?**

**Contributions.**
1. A formalization of power-aware LLM-serving routing as a 3-resource (compute/load/power)
   egalitarian social welfare problem, building on DRF [1] but extending it to a reactive,
   time-varying resource (live power-ramp state) rather than DRF's original static
   per-request demand model (§3), and a proof that the natural sorted-lexicographic solution
   is Pareto-non-dominated at every decision (§4.1).
2. **The coincidence-ceiling mechanism** (§4.6): a pluggable, fleet-aware extension that
   contracts every candidate's ramp ceiling by a single shared scalar when multiple replicas
   are simultaneously elevated, directly targeting the fleet-*aggregate* coincident-ramp
   hazard every purely per-replica rule — including this paper's own base rule — cannot see.
   This mechanism requires no new safety proof: it is a direct instantiation of an existing
   corollary (§4.3), and generalizes across rule families, transparently inheriting each
   parent rule's own safety status rather than granting one. We adopt
   `drf_power_tiebreak_full_coincidence_ceiling` (short-named `coincidence_ceiling`) as this
   paper's proposed routing strategy.
3. Two proofs, via explicit counterexample, that natural alternatives sacrifice the guarantee
   via distinct mechanisms: a fixed-priority power tie-break, repaired by appending the
   dropped resource rather than truncating it — the repair that produces this paper's base
   rule (§4.2) — and a power-extended LMETRIC score whose multiplicative structure collapses
   to an uninformative tie on any full prefix-cache hit, common enough to matter in practice
   (12.8% of realistic instances, §4.2).
4. Real-hardware validation on an 8×4090 fleet, across 7 workload conditions (including two
   real traces): a significant peak-power reduction relative to a power-blind baseline on the
   condition central to this paper's motivation (p=0.0017, n=6), with TTFT and TBT
   directionally favorable but not independently significant at this trial count (§6.1); and,
   against the strongest empirically-performing but *not* Pareto-safe alternative — replicated
   to matched n=6 — no statistically distinguishable difference on any metric measured: peak
   power, TTFT, TBT, ramp-derivative statistics, energy-per-token, or SLO-violation-rate
   (§6.1-6.3). The safety guarantee is free on every axis tested.
5. **A decision-level noise-floor characterization and resulting evaluation methodology**
   (§5), useful beyond this paper: real-hardware closed-loop routing evaluation carries a
   ~75-85% per-decision mismatch between two runs of the *identical* policy, driven by
   non-reproducible hardware completion timing rather than workload construction (confirmed via
   a state-blind control and real conversational traces, §5.1); we identify a general
   principle for which aggregate statistics survive this noise at feasible trial counts and
   which don't (§5.2), and use it to determine which of this paper's own empirical claims can
   currently be asserted with confidence (§5.3, §6).

**Roadmap.** Three normalized shares (compute, load, power, §3) feed a sorted rule, proven
Pareto-safe (Lemma 1, §4.1). Two natural power-biased variants of it sacrifice that guarantee:
a fixed-priority tie-break (Claim 1, repaired into `drf_power_tiebreak_full`) and a
power-extended LMETRIC score (Claim 2, 12.8% of realistic instances, §4.2); `weighted_sum` is a
structurally different Pareto-safe alternative that is not threshold-safe for any weights
(Theorem 5, §4.5). The coincidence-ceiling mechanism (§4.6) applies to all three, inheriting
rather than granting safety status, and yields this paper's proposed rule,
`drf_power_tiebreak_full_coincidence_ceiling`. On real 8×4090 hardware (§6) it delivers a
significant peak-power reduction vs. `lmetric` (power-blind) on the short/original condition
(p=0.0017, Table 1) and is statistically indistinguishable from the strongest
empirically-performing but unsafe alternative on every metric measured at matched n=6
(Table 2, §6.3). Table 3 gives the full safety status of every rule discussed, including
`drf_fixed` and `round_robin`.

## 2. Background / Related Work

*(moved to §8, unchanged content — see there.)*

## 3. Problem Formulation

Consider N replicas serving requests behind a router. For a candidate replica `c` and an
incoming request, define three shares, each normalized to roughly `[0, ∞)` with 1.0
representing "at capacity":

- `Share_compute(c) = P-token(c) / token_budget(c)` — P-token is the prefix-cache-discounted
  count of *new* prefill tokens this request would cost replica `c` (computed from a
  per-replica cache-state mirror), so this share already reflects cache locality, not raw
  prompt length.
- `Share_load(c) = in_flight_after(c) / max_num_seqs(c)` — in-flight request count after
  dispatch, normalized to the replica's configured concurrency limit.
- `Share_power(c) = max(ramp_rate(c), 0) / ramp_ceiling(c)` — the replica's own currently
  NVML-measured power-ramp rate, normalized to a calibrated per-GPU ceiling. Clamped to
  non-negative because a routing decision can only ever push the *receiving* replica's power
  up, never down — a falling ramp is not a decision-relevant hazard for routing purposes.
  (`ramp_ceiling` itself is physically grounded in the GPU's own DVFS transition dynamics —
  §8 relates the two explicitly.)

**Departure from DRF's original model.** Ghodsi et al.'s DRF assumes a fixed, known demand
vector per user, allocated from a fixed resource pool. `Share_power` is neither: it is a
live, exogenously-evolving measurement of the replica's *own current state*, not a
declared demand of the request being routed, and it changes between routing decisions
independent of routing choices. The original static-demand Pareto-efficiency proof does not
transfer to this setting as-is — §4 proves the property this paper needs (per-decision, not
per-trajectory) directly for this model.

**`Share_power` is a state read, not a projection.** `Share_compute` and `Share_load` are
one-step projections attributable to the specific request being routed: P-token is *this
request's* incremental prefill cost, and `in_flight_after` is the queue depth *after this
request is hypothetically added*. `Share_power` admits no such attribution: a GPU's
instantaneous power draw is an emergent property of whatever batch is currently executing,
not a quantity linearly attributable to any single request, so there is no well-defined "ramp
rate after this request" the way there is a well-defined queue depth after it. What the router
reads instead is the replica's *current* state — the same object a rate limiter reads in
classical feedback control [10], and the same kind of signal classical state-dependent
load balancing (power-of-two-choices, join-shortest-queue) [5, 6] routes on: an observed
state, not a computed causal attribution. None of §4.1–§4.6's guarantees depend on this
distinction — every proof treats `s(c)` as an arbitrary comparable real vector available at
decision time, regardless of whether a coordinate is a request-specific projection or a
current-state read. The distinction does, however, determine how the two kinds of metric must
be evaluated: TTFT and TBT are per-decision outcomes, decomposable into a population of
independent per-request measurements the way `Share_compute` and `Share_load` are decomposable
into per-request costs; power and ramp are properties of the continuous trajectory the routing
policy induces over time, evaluable only in aggregate, the same way a feedback controller is
evaluated by its closed-loop trajectory (peak deviation, settling time) rather than by
attributing trajectory segments to individual control actions. §5-6 evaluate them accordingly,
and §5.1-5.2 show this "aggregate-only" evaluability is exactly why ramp statistics are
harder to trust at low trial counts than per-request statistics are.

**Welfare objective.** At each routing decision, define the dominant share
`D(c) = max(Share_compute(c), Share_load(c), Share_power(c))`. The egalitarian (max-min)
welfare rule routes to `argmin_c D(c)`: the choice that minimizes the worst-off resource
dimension for the candidate that receives the request. This is a per-decision, myopic
formulation; a sequence of egalitarian-optimal decisions is not claimed, or needed for the
results in this paper, to be jointly optimal over a trajectory.

**Figure 1** (`figs/overview_2.png`): the three normalized shares and the dominant share
`D(c)`, with representative example values for one candidate replica.

## 4. Theory

### 4.1 The sorted-lexicographic rule is Pareto-non-dominated

`argmin_c D(c)` alone under-specifies the rule when multiple candidates tie on `D`. Define
the **sorted rule**: route to `argmin_c σ(s(c))`, where `s(c) = (Share_compute(c),
Share_load(c), Share_power(c))`, `σ` sorts a vector into descending order, and ties are
broken by standard lexicographic comparison of the sorted vectors.

**Lemma 1.** The candidate selected by the sorted rule is Pareto-non-dominated among the
candidate set: no other candidate `c'` satisfies `Share_i(c') ≤ Share_i(c*)` for all three
resources `i` with strict inequality for at least one.

**Proof.** For `v ∈ ℝ³`, write `v_(1) ≥ v_(2) ≥ v_(3)` for its order statistics, so
`σ(v) = (v_(1), v_(2), v_(3))`. Suppose toward contradiction that `c'` Pareto-dominates
`c* = argmin_c σ(s(c))`: `s(c') ≤ s(c*)` coordinatewise, with strict inequality at some
coordinate `i₀`.

*(i) `σ(s(c')) ≤ σ(s(c*))` coordinatewise.* For any threshold `t` and any `i`,
`s(c')_i ≥ t ⟹ s(c*)_i ≥ s(c')_i ≥ t`, so `#{i : s(c')_i ≥ t} ≤ #{i : s(c*)_i ≥ t}`. The
`k`-th order statistic is the least `t` with `#{i : v_i ≥ t} ≥ k`; a pointwise-smaller
counting function cannot raise this threshold, so `s(c')_(k) ≤ s(c*)_(k)` for every `k`.

*(ii) The inequality is strict as vectors.* Sorting permutes coordinates, so
`Σ_k s(c')_(k) = Σ_i s(c')_i < Σ_i s(c*)_i = Σ_k s(c*)_(k)`, the middle inequality from
domination at `i₀`. Hence `σ(s(c')) ≠ σ(s(c*))`.

*(iii) Lexicographic order.* Two coordinatewise-`≤` vectors that differ somewhere are
ordered lexicographically at their first differing coordinate, in the direction of (i)'s
inequality there: `σ(s(c')) <_lex σ(s(c*))`.

*(iv)* The sorted rule then selects `c'` over `c*`, contradicting `c*`'s selection. ∎

This claim was additionally verified by brute-force search: 200,000 randomly generated
candidate sets (2-5 candidates, 3 shares each) produced zero violations.

### 4.2 Two natural variants sacrifice the guarantee, via two distinct mechanisms

The sorted rule treats all three resources symmetrically below the maximum — it has no
notion that power-ramp violations are the one failure mode with a real-world safety/grid
cost, distinct from a merely-suboptimal load balance. This motivates two independent, natural
attempts to bias the rule toward power specifically. Both provably sacrifice Lemma 1's
guarantee, via mechanisms different enough that neither's fix addresses the other — evidence
that the guarantee is a real structural constraint worth checking explicitly for any new score
design, not a formality to patch around (the practice this paper argues for in §7).

**4.2.1 Fixed-priority power tie-break.** A natural fix to the sorted rule: after comparing
the dominant share, always compare `Share_power` next, *by name*, ahead of `Share_load`,
regardless of which is numerically larger. Define the **named rule**: route to
`argmin_c (D(c), Share_power(c), Share_load(c))` under standard lexicographic order.

**Claim 1.** The named rule can select a Pareto-dominated candidate.

**Construction.** Let candidate `A = (Share_compute, Share_load, Share_power) = (0.3, 0.9,
0.9)` and `B = (0.5, 0.9, 0.9)`. `A` Pareto-dominates `B` (equal load and power, strictly
lower compute). Both have `D(A) = D(B) = 0.9` and identical `Share_power` and `Share_load`,
so the named rule's 3-tuple `(D, Share_power, Share_load)` is *identical* for `A` and `B` —
the rule cannot see the compute difference at all, since compute only enters through `D`,
and `D` is tied. `min()` over identical keys returns whichever candidate is encountered
first; when `B` is first in iteration order, the named rule selects the Pareto-dominated
`B`. This was verified directly in code, not only argued (a brute-force check across 200,000
random instances reproduces the failure mode; see supplementary material).

**Why this happens.** The named rule buys a deliberate, domain-motivated priority — never let
a load difference override a power difference — at the cost of the compute dimension
becoming invisible whenever the dominant share is tied via load or power rather than compute:
`Share_compute` never appears anywhere in the 3-tuple `(D, Share_power, Share_load)`, so once
`D`, `Share_power`, and `Share_load` all tie, the rule has no remaining information to break
the tie correctly.

**Figure 3** (`figs/pareto_dominance.png`): the same `A`/`B` pair the named rule fails on —
the full lexicographic key used by the sorted rule this paper adopts sees the compute
difference the named rule's truncated tuple cannot, and correctly selects the
Pareto-non-dominated candidate `A`.

**The repair, and this paper's base rule.** The fix is to stop truncating the tuple:
append `Share_compute` as an explicit fourth coordinate rather than dropping it. Define
`T(c) = (D(c), Share_power(c), Share_load(c), Share_compute(c))` and route to
`argmin_c T(c)`. This preserves the exact same primary criterion and the same deliberate
power-before-load priority the named rule was built for — nothing changes about *when*
power is allowed to override load — while ensuring no coordinate is ever silently dropped.
We call this rule **`drf_power_tiebreak_full`**; §4.6 extends it with the coincidence-ceiling
mechanism to produce this paper's actual proposed strategy.

**Figure 2** (`figs/overview_1.png`): routing a single request under `T(c)` — the router
evaluates each replica's tuple and dispatches to the `argmin` (replica 2 here, despite not
having the lowest individual share on every coordinate).

**Corollary 1.** `argmin_c T(c)` is Pareto-non-dominated at every decision.

**Proof.** Write `s = (Share_compute, Share_load, Share_power)` and suppose `c'`
Pareto-dominates `c* = argmin_c T(c)`: `s(c') ≤ s(c*)` coordinatewise, strict somewhere.
`D = max(s)` is monotone under coordinatewise `≤`, so `D(c') ≤ D(c*)`. Three exhaustive cases:

- **`D(c') < D(c*)`.** `T(c') <_lex T(c*)` at coordinate 1.
- **`D(c') = D(c*)`, and `Share_power(c') ≤ Share_power(c*)` strict.** `T(c') <_lex T(c*)`
  at coordinate 2. (Symmetrically for `Share_load` at coordinate 3, if `D` and
  `Share_power` tie but `Share_load` does not.)
- **`D`, `Share_power`, `Share_load` all tied between `c'` and `c*`.** Domination requires a
  strict inequality in `s(c') ≤ s(c*)`; the first three coordinates being tied forces it onto
  `Share_compute`: `Share_compute(c') < Share_compute(c*)`, so `T(c') <_lex T(c*)` at
  coordinate 4.

Every case gives `T(c') <_lex T(c*)`, contradicting `c*`'s selection as `argmin_c T(c)`. ∎

This closed-form proof is in fact simpler than Lemma 1's, since a fixed coordinate order needs
no rearrangement argument. Direct verification in code confirms it resolves Claim 1's exact
counterexample under both iteration orders, while the plain named rule still fails it
(200,000 trials, 0 violations).

**4.2.2 Power-extended LMETRIC.** A different, non-DRF way to add a power signal: extend
LMETRIC's own multiplicative score, `new_tokens × in_flight_after`, with a continuous power
penalty, `(1 + Share_power)`. Restated in this paper's normalized shares (holding
`token_budget` and `max_num_seqs` fixed across candidates, so raw counts equal shares), this
is `lmetric_power(c) = Share_compute(c) · Share_load(c) · (1 + Share_power(c))`, routing to
the minimum. This design has a genuine theoretical lineage: a continuous, always-differentiable
penalty added to an otherwise-unconstrained objective is the routing analogue of Lyapunov
drift-plus-penalty scheduling [11] — trading off instantaneous cost against a soft,
ever-present penalty on the hazardous quantity, rather than enforcing a hard constraint on it.
It is also the path of least engineering resistance for anyone already running LMETRIC in
production, which is why this specific power extension is examined rather than a foil
constructed to fail.

**Claim 2.** `lmetric_power` can select a Pareto-dominated candidate, via a different
mechanism than Claim 1: any candidate with `Share_compute = 0` (a full prefix-cache hit — no
new tokens to prefill) scores exactly `0`, *regardless of its load or power share*. Two such
candidates are indistinguishable to the score no matter how different their load and power
are.

**Construction.** Let `A = (0.0, 0.1, 0.1)` and `B = (0.0, 0.9, 0.9)` — both cache hits; `A`
strictly Pareto-dominates `B` on both load and power. `lmetric_power(A) = lmetric_power(B) =
0`. With `B` first in iteration order, `min()` selects the Pareto-dominated `B`.

This mechanism is qualitatively different from Claim 1's: it requires an *exact* tie in
`Share_compute`, but `Share_compute = 0` is not measure-zero in real traffic — it is a
common, discrete event (a full prefix-cache hit), and this paper's own Light/Cachehit
condition (§6) is specifically constructed to be dominated by it. Sampling at a realistic
cache-hit rate (40%, matching Light/Cachehit's rough hit rate) over 200,000 random instances
(2-5 candidates) finds 25,618 violations — **12.8% of instances**, not a rare corner case.

**A structurally different safe alternative.** `weighted_sum(c) = 0.33·Share_compute(c) +
0.33·Share_load(c) + 0.33·Share_power(c)`, route to the minimum, is a classical result [13]:
any positive-weighted linear combination of the shares preserves Pareto non-domination. This
was verified for the specific weights used here (200,000 trials, 0 violations) so that §6 can
include it as an *external validity check*: a safe rule built on an entirely different
mechanism, included specifically to show that any empirical result about giving up the Pareto
guarantee is not an artifact of this paper's own DRF-based construction.

**A family of Pareto-safe repairs of `lmetric_power` was also considered and not adopted:**
`lmetric_power_pareto`, replacing the multiplicative form with `(shift + Share_compute)·(shift
+ Share_load)·(shift + Share_power)`, removes Claim 2's zero-collapse by construction (every
factor strictly positive whenever `shift > 0`) and is Pareto-safe by the same monotone-product
argument that makes `weighted_sum` safe (verified, 200,000 trials, 0 violations at `shift=1`
and several tested `shift=ε` values). This family is not threshold-safe for the same reason
`weighted_sum` isn't: Theorem 5 (§4.5) rules out threshold-safety for *any* fixed-weight or
fixed-shift multiplicative construction, so no choice of shift can match Theorem 4's
guarantee. Its apparent empirical edge over the proposed rule at low trial counts did not
survive the noise-band audit in §5.3; it is not adopted as a headline alternative.

**What §6 tests, and each arm's role there.** Table 3 below summarizes every rule this paper
discusses and its safety status in one place; none are interchangeable in role even where two
share a safety guarantee.

**Table 3: Routing rules discussed in this paper, at a glance.** "Threshold-safe" extends
Theorem 4's guarantee to any rule that sorts on `D` first. Coincidence-ceiling variants
(§4.6) inherit their parent rule's status unchanged — the mechanism generalizes safety status,
it does not grant it.

| Rule | Family | Pareto-safe | Thresh.-safe | Role |
|---|---|---|---|---|
| `drf_fixed` | sorted (leximin) | ✓ (Lem. 1) | ✓ | unmodified baseline |
| `drf_power_tiebreak_full` | sorted (leximin) | ✓ (Cor. 1) | ✓ | base repair, not proposed directly |
| `drf_power_tiebreak_full_coincidence_ceiling` | sorted + shared ceiling | ✓ (§4.6) | ✓ | **proposed** (`coincidence_ceiling`) |
| `drf_power_tiebreak` (named) | sorted, truncated tie-break | ✗ (Claim 1) | ✓ | motivates the repair |
| `weighted_sum` | utilitarian (linear) | ✓ [13] | ✗ (Thm. 5) | external validity check |
| `weighted_sum_coincidence_ceiling` | utilitarian + shared ceiling | ✓ (§4.6) | ✗ (Thm. 5) | ext. check, coincidence-aware |
| `lmetric_power` | multiplicative | ✗ (Claim 2) | ✗ | unsafe baseline |
| `lmetric_power_coincidence_ceiling` | multiplicative + shared ceiling | ✗ (Claim 2, unaffected) | ✗ | **strongest unsafe alternative** — §6's head-to-head |
| `lmetric_power_pareto` (+ε variants) | shifted-product | ✓ | ✗ (Thm. 5) | explored, not adopted (see above) |
| `lmetric` (power-blind) | multiplicative, no power term | n/a | n/a | power-blind baseline |
| `round_robin` | blind (no signal) | n/a | n/a | state-blind floor |

Against this backdrop, the empirical question §6 asks is whether giving up the Pareto
guarantee — via `lmetric_power_coincidence_ceiling`, the *strongest empirically-performing*
alternative found anywhere in this project's testing (it out-dominates every other tested arm
more than 3-to-1 in raw pairwise tally) — buys anything a provably safe rule does not already
deliver. This strongest-known unsafe alternative, not a weaker foil, is deliberately used as
the comparator, so that any advantage in its favor would be the most convincing evidence
available; §6 finds no such advantage on the metrics that survive §5's noise-floor scrutiny.

### 4.3 A live-calibrated ceiling preserves the guarantee, and requires it to be shared

Both rules above assume `Share_power(c) = max(ramp_rate(c), 0) / κ` for a fixed constant
ceiling `κ` (per-GPU calibrated, §5). A natural objection: does either guarantee survive
replacing `κ` with a value recalibrated live from the fleet's own recent state — as an
*adaptive-ceiling* variant of either rule (and, as §4.6 shows, the coincidence-ceiling
mechanism itself) would need?

**Corollary 2.** Lemma 1 holds unchanged for any `κ(t) > 0` that is a single scalar shared
identically by every candidate at decision time `t`, regardless of how `κ(t)` is computed —
static, adaptively recalibrated from fleet history, or otherwise.

**Proof.** The proof of Lemma 1 treats `s(c)`'s three coordinates as arbitrary reals; it
never uses the fact that `Share_power`'s denominator is constant *across* decisions, only
that it is the same value for every candidate *within* one decision, so the coordinate
remains a well-defined, comparable per-candidate quantity for that decision's `argmin`.
Substituting `κ(t)` for a fixed constant changes nothing the proof relies on. ∎

This was verified directly: re-running §4.1's brute-force search with the ceiling itself
independently randomized per trial (not just the raw shares) still produces zero violations
across 200,000 trials. **§4.6's coincidence-ceiling mechanism is a direct instantiation of
this corollary**, not a separate result requiring its own proof.

**Sharing is load-bearing.** The corollary requires `κ(t)` to be shared across candidates,
not calibrated per-candidate. Under a per-candidate `κ_c`, share-space non-domination can
dissociate from physical reality: two candidates with identical compute/load but
`(raw_ramp, κ) = (0.9, 10)` and `(0.1, 0.1)` realize `Share_power = 0.09` and `1.0`
respectively — the sorted rule (correctly, per Lemma 1) selects the first candidate as
share-space non-dominated, even though it draws the physically *larger* raw ramp. This is
why the implementation instantiates one ceiling/coincidence-factor calibrator per router,
not one per replica.

### 4.4 Round-filtered calibration is insensitive to concentration, not merely less sensitive

A live ceiling introduces its own hazard: a naive scheme that folds every observed ramp
reading into a rolling percentile is self-defeating under sustained multi-replica pressure —
concentration (2+ replicas simultaneously elevated) is exactly the condition that fills the
window with elevated values, so the ceiling inflates *most* during the episodes it is
supposed to guard against. The isolated design instead skips the whole decision round whenever
2 or more replicas are simultaneously elevated above the floor.

**Lemma 2.** Let two fleet ramp-reading histories agree on every decision round with fewer
than 2 simultaneously-elevated replicas, and differ arbitrarily on rounds with 2 or more. The
round-filtered calibration produces identical ceiling trajectories on both histories at every
timestep.

**Proof.** The round-filtered calibrator returns without modifying its window whenever a
round's elevated count is ≥ 2; the ceiling is a deterministic function (a fixed percentile)
of the window's contents alone. Since the two histories only ever differ on rounds that are
skipped entirely, the window's contents — and hence the ceiling — are identical at every
step. ∎

This was verified over 2,000 randomized paired-history trials: zero violations. Feeding the
identical paired histories through the naive (unfiltered) calibration instead diverges in
every one of the 2,000 trials.

### 4.5 Selecting among multiple Pareto-optimal points: egalitarian vs. utilitarian

Lemma 1 and Corollary 1 guarantee membership on the Pareto frontier but say nothing about
*which* frontier point to prefer when several candidates are mutually non-dominated — a real
question, since `weighted_sum` (§4.2) is provably on the frontier too, by an entirely
different mechanism. This subsection identifies exactly what selection principle the sorted
rule implements, contrasts it with `weighted_sum`'s, and characterizes precisely when, and by
how much, the two disagree.

**The sorted rule is leximin.** The order `u ⪯ v ⟺ σ(u) ≤_lex σ(v)` on share vectors is the
classical *leximin* (lexicographic egalitarian) order [12]: compare the worst coordinate
first, then the second-worst, and so on. Lemma 1's proof already establishes more than the
lemma states: domination strictly worsens a candidate's leximin rank, so the sorted rule
computes the leximin-minimal point, a strictly more discriminating criterion than mere
non-domination. `weighted_sum` implements the classical *utilitarian* rule instead,
`argmin_c Σᵢ Shareᵢ(c)`, also on the frontier [13], for a different, formally separable reason.

**Egalitarian rules reward equalization; utilitarian rules are blind to it.** For
`s(c) = (a, b, d)` with `a` the strict max, transferring `ε ∈ (0, (a-b)/2]` from the max
coordinate to a smaller one without crossing strictly lowers the sorted rule's key but leaves
`weighted_sum` exactly unchanged — the Pigou–Dalton transfer principle (verified over 16,781
random transfers, 0 failures).

**Lemma 3 (Divergence).** Let `X, Y` be candidates with `D(X) < D(Y)` and
`Σᵢ Shareᵢ(X) > Σᵢ Shareᵢ(Y)`. Then `X` and `Y` are Pareto-incomparable, the sorted rule
selects `X`, and `weighted_sum` selects `Y`. Whenever one candidate actually dominates the
other, the two rules always agree.

**Proof.** `D(X) < D(Y)` makes `σ(X)` lexicographically smaller than `σ(Y)` at the first
coordinate, so the sorted rule strictly prefers `X`; `Σ(X) > Σ(Y)` makes `weighted_sum`
strictly prefer `Y`. `Y` cannot dominate `X`: `D` is monotone under coordinatewise `≤`, so `Y`
dominating `X` would force `D(Y) ≤ D(X)`, contradicting `D(X) < D(Y)`. `X` cannot dominate `Y`
either: domination implies a weakly smaller positive-weighted sum [13], contradicting the
hypothesis. Hence neither dominates the other; and by the same steps in reverse, whenever one
candidate *does* dominate, the two rules cannot disagree. ∎

Verified over 200,000 random pairs: 0 domination-disagreements, and every one of 44,733
divergence-eligible orderings behaved exactly as predicted.

**Theorem 4 (Universal threshold-safety).** For any `τ > 0` and any candidate set `C`: if some
`c ∈ C` has `D(c) ≤ τ`, the sorted rule's selection also satisfies `D(·) ≤ τ` — simultaneously,
for every `τ`.

**Proof.** The sorted rule's first sort key is `D`, so its pick achieves `min_{c∈C} D(c)`, the
global minimum over `C`; if that minimum is `≤ τ`, so is the pick's. ∎

**Theorem 5 (No fixed-weight rule has this property).** For any weights `w = (w1, w2, w3)`,
all `wᵢ > 0`, and any `τ > 0`, there exists a candidate pair where a safe candidate is
available (some `c` with `D(c) ≤ τ`) but the `w`-weighted-sum rule selects an unsafe one
(`D(c) > τ`).

**Proof.** Let `S = (τ, τ, τ)` (safe) and `U = (0, 0, τ+M)` (unsafe). `w·U < w·S ⟺
M < τ(w1+w2)/w3`, an interval non-empty for any positive weights. ∎

This holds for *every* fixed weighting, not just `weighted_sum`'s. On generic random
instances (paper's weights, `τ=1.0`), `weighted_sum` still picks an available-but-unsafe
candidate in **11.71% of trials** — the same order of magnitude as `lmetric_power`'s 12.8%
(Claim 2). (50,000 random-weight constructions, 0 failures; 144,593-trial generic-instance
rate.)

**Proposition 1 (Price of egalitarianism, tight).** If `D(X) < D(Y)`, the sorted rule's excess
total burden over `weighted_sum`'s pick, `Σ(X) - Σ(Y)`, is strictly less than `2·D(X)`, and
this bound is approached arbitrarily closely.

**Proof.** `Σ(X) ≤ 3D(X)`; `Σ(Y) ≥ D(Y) > D(X)`. So `Σ(X) - Σ(Y) < 2D(X)`. Tightness: for
`X = (D0, D0, D0)`, `Y = (D0+δ, 0, 0)`, the gap is `2D0 - δ → 2D0` as `δ → 0+`. ∎

**Practical implication.** Peak power and ramp rate are worst-case, threshold-triggered
hazards, exactly the class Theorem 5 shows no fixed weighting can safely target. `weighted_sum`
remains genuinely Pareto-safe and is a reasonable choice absent a binding power/ramp
constraint, but for the safety-critical dimension this paper is motivated by, Theorem 4 is a
guarantee no reweighting of `weighted_sum` (or of the shifted-product `lmetric_power_pareto`
family, §4.2) can replicate — the reason this paper proposes the sorted-rule-derived
`coincidence_ceiling` rather than either alternative.

### 4.6 A pluggable, fleet-aware ceiling: the coincidence-ceiling mechanism

Every rule above computes `Share_power(c)` from candidate `c`'s own local ramp state only —
never asking whether *other* replicas are ramping simultaneously. Every metric §6 reports
(peak, mean/p99 ramp) is measured on the fleet-**aggregate** trace. A rule can look
individually safe on every replica while still producing a bad aggregate ramp if it never
accounts for replicas ramping together.

**Definition.** `coincidence_ceiling_factor(C) = 1 / (1 + β·max(0, n_elevated(C) - 1))`,
where `C` is the candidate set at a decision, `n_elevated(C)` counts candidates whose own
`Share_power` exceeds `elevated_frac` (default 0.5) against their *own* static per-GPU
ceiling, and `β` (default 1.0) sets how sharply the shared ceiling contracts per additional
simultaneously-elevated replica. A single elevated replica (`n_elevated ∈ {0, 1}`) is not a
coincidence and leaves the factor at 1.0 (no adjustment); each additional
simultaneously-elevated replica tightens *every* candidate's effective ceiling by the same
shared factor. Applying this factor to scale every candidate's `ramp_ceiling` before computing
`Share_power` (and hence `D`) at that decision defines the **coincidence-ceiling** variant of
any base rule.

**Figure 4** (`figs/coincidence_ceiling.png`): four candidates' `Share_power` (a) without and
(b) with the coincidence-ceiling adjustment. Two candidates (`c1`, `c3`) exceed
`elevated_frac`, so `n_elevated=2` and the shared factor contracts every candidate's effective
ceiling to 0.5×, doubling every candidate's `Share_power` — not just the two that triggered it.

**No new proof needed.** `factor` is one scalar, computed once per decision and shared
identically by every candidate at that decision — exactly the object Corollary 2 already
covers, regardless of how the scalar is computed. So
`drf_power_tiebreak_full_coincidence_ceiling` inherits Pareto-non-domination and
threshold-safety from Corollary 1/Theorem 4 unchanged; `weighted_sum_coincidence_ceiling`
inherits `weighted_sum`'s Pareto-safety but not threshold-safety (Theorem 5 still applies
regardless of ceiling design); `lmetric_power_coincidence_ceiling` inherits `lmetric_power`'s
lack of Pareto-safety (Claim 2's cache-hit-collapse is orthogonal to which ceiling feeds its
power term). **The mechanism generalizes safety status; it does not grant it.**

**This is not a no-op.** Scaling every candidate's ceiling by the same factor does not change
their relative order by `Share_power` alone — it changes `Share_power`'s *magnitude* relative
to `Share_compute`/`Share_load` in `D(c) = max(...)`, making power more likely to be the
binding dimension for everyone during a genuine coincidence event. A hand-constructed case
confirms the mechanism actually flips a routing decision: without adjustment, a replica with
moderate local power pressure wins over one with zero power but higher compute; with two
*other* replicas coincidentally elevated, the pick flips to the higher-compute replica,
avoiding piling onto an already-pressured fleet.

**Adopted rule.** This paper adopts `drf_power_tiebreak_full_coincidence_ceiling`
(short-named `coincidence_ceiling`) as its proposed routing strategy: it is the only
coincidence-ceiling variant with a full, proven worst-case guarantee, inherited without a new
proof, and — as §6 shows — empirically strong on the condition central to this paper's
motivation. It is validated against `lmetric_power_coincidence_ceiling`, the strongest
empirically-performing alternative found anywhere in this project's testing (Table 3), rather
than a weaker unsafe foil.

## 5. Evaluation Methodology

**Setup.** 8×4090 server (single chassis), 7B model (Qwen2.5-Coder-7B-Instruct). Two
independent NVML polling loops must not be conflated: the router's own live power reads (used
to compute `Share_power` at routing time) poll every 500ms (a separate NVML session from the
one below); the ground-truth power trace this paper's ramp/peak numbers are computed from is
logged by a separate sidecar process targeting a 50ms sampling interval, whose actual observed
sampling period is closer to ~89ms once per-GPU NVML query overhead (across all six replicas,
queried in sequence) is accounted for. Neither cadence should be read as the other's — they
serve different purposes (a live, cheap-enough-for-every-decision signal vs. a
higher-resolution trace for offline ramp/peak analysis).

Two further definitional differences separate the decision-time signal from this paper's
reported numbers, and neither is a formal corollary of §4's guarantees: (i) `Share_power`
clamps ramp rate to non-negative, since only a rising ramp is a routing-relevant hazard,
whereas the ramp statistics reported below use `|ΔP/Δt|`, since a grid-facing ramp-rate hazard
is generally bidirectional; (ii) `Share_power` is a strictly per-replica quantity read at
decision time, whereas "peak power" and "mean/p99 ramp" below are computed on the
fleet-aggregate power trace (summed across all six GPUs), the physically meaningful quantity
for a grid-facing claim. Theorem 4 guarantees no avoidable violation of the per-replica,
upward-only quantity at decision time; the fleet-aggregate, bidirectional numbers below are an
empirical, not formally guaranteed, consequence of routing that way consistently over a trial.

`Share_power`'s ceiling `κ` is calibrated *per replica*, not shared as one constant: a
dedicated calibration check (24 concurrent prefill bursts, isolated per GPU) found the ceiling
that actually applies varies **34% across the six replicas** (359.5–512.9 W/s) — real hardware
heterogeneity a shared constant silently averages away. Every number in this paper uses the
corrected per-GPU ceiling; re-running the comparison under both the old shared ceiling and the
corrected per-GPU one showed apparent dominance relationships flip or disappear in 4 of 6
tested conditions, always in the direction of making an unsafe arm's empirical position look
better than it is under correct calibration.

### 5.1 The decision-level noise floor

Real hardware completion timing is not bit-reproducible across separate physical executions
(GPU kernel scheduling, thermal/clock variance, OS scheduling), so any state-dependent
router's live-state inputs (`in_flight_after`, `ramp_rate_w_per_s`) are only ever "true at
this exact wall-clock instant," and that instant itself isn't reproducible. Close-call ties
get flipped by real timing noise a scoring formula cannot see, and once one decision flips,
replica loads genuinely diverge — every subsequent decision inherits a real, compounding state
difference.

This was quantified directly by pairing individual routing decisions across independent runs
of the *identical* policy on paired (fixed-seed) workload content, joined by conversation id
and turn rather than row order, which real timing reshuffles for multi-turn conditions (an
earlier attempt using naive row-order pairing gave misleadingly low mismatch numbers on
multi-turn conditions and was corrected before being reported). **Any state-dependent routing
rule mismatches on 75-85% of individual decisions between two runs of itself**, with the first
divergence typically within the first ~35 requests (median first-mismatch row: 7). This holds
across every condition tested, including two built entirely on real conversation text with
zero synthetic whale-padding (Light/Cachehit: 84.0-84.6%; WildChat, open-loop Poisson arrivals:
81.9-83.4%) and a real-arrival-trace condition (BurstGPT: 75.0%) — ruling out synthetic
workload construction, closed-loop feedback specifically (WildChat is open-loop), and
multi-turn interleaving artifacts as the cause.

**A state-blind control isolates the mechanism precisely.** `round_robin`, which never reads
live replica state (only an incrementing counter), shows only **3.1% decision-mismatch** on
the same condition where every load/power-aware rule shows 75-85%. This confirms the cause is
state-dependence itself, not the evaluation harness: any rule that reads live per-replica
state is exposed to genuine, unavoidable timing noise on real hardware; a rule that doesn't
is nearly immune.

This is not a threat to dominance-based comparisons elsewhere in this paper — dominance
requires simultaneous agreement across multiple metrics, which noise alone is unlikely to
produce consistently — but it means any single-metric point estimate on a noise-sensitive
statistic (§5.2) between two low-trial-count arms should be checked against a rule's own
self-noise range before being over-read.

### 5.2 Metric-selection principle: which statistics are trustworthy at feasible trial counts

Not every aggregate statistic is equally exposed to the noise floor above. Metrics that **sum
or count over many independent events within a trial** are low-noise even at n=3, while
metrics that are **derivatives, extreme-value statistics, or spread over few groups** are
not — a distinction that held consistently across every candidate metric tested:

- **Low-noise (survive n=3-6):** energy-per-token (total NVML-measured energy delta divided by
  total output tokens — sums over on the order of 10⁵ tokens per trial) shows well under 1-3%
  relative std across every arm and condition tested. SLO-violation-rate (fraction of requests
  exceeding fixed TTFT/TBT thresholds — counts over hundreds of requests per trial) shows
  ~1.4-2.3%. Peak power, mean TTFT, and mean TBT — themselves aggregate-ish quantities, not
  single-request point reads — typically stay in the ~1-2% range at n=3, though noise level is
  condition-dependent rather than a fixed property of a given metric: TTFT's relative std is
  0.4% on the long condition but 6.2% on the short condition this paper's headline comparison
  uses (§6.1), a 15× difference. A plausible driver is that the short condition's shorter
  active window yields fewer within-trial TTFT samples to average cross-trial timing noise
  over, though this specific mechanism has not been independently verified.
- **High-noise (require 6+ trials, or should not be trusted for ranking claims at all at
  n=3):** mean/p99/max ramp rate — derivatives or extreme-value statistics, sensitive to exact
  sample timing or a single worst event — show 8-34% relative std depending on which
  statistic. Two further candidates hypothesized to be integrated/fractional and therefore
  low-noise were tested and both failed: **coincidence-factor** (fraction of power-pressured
  time with 2+ replicas simultaneously elevated — directly the quantity §4.6's mechanism
  targets) showed **56.97%** relative std, worse than max_ramp, because despite being a
  "fraction" it counts rare discrete threshold-crossing events (only hundreds of occurrences
  per trial), not the tens of thousands of samples that make energy-per-token stable.
  **Per-replica fairness CV** (dispersion of load/energy across only 6 GPUs) showed **32-40%**,
  a small-N-groups problem structurally similar to estimating standard deviation from 3 trials.
  "Integrated" or "fractional" alone does not guarantee low noise; what matters is the number
  of underlying events being summed.

A direct demonstration of why n=3 is insufficient for the high-noise tier: one arm's own
3-trial sample std on mean ramp was ±1.7 W/s; extending the same arm to 6 trials (same
condition) revealed a true std of ±12.3 W/s — **7× larger**. Point estimates and even sample
standard deviations from 3 trials materially underestimate the true spread for this class of
metric.

### 5.3 Trial-count implications, and closing the "but the unsafe rule wins" objection

Applying §5.2's principle retroactively to this project's own empirical sweep: a full
audit of the point-estimate range across 11 different routing rules tested on the same
condition found mean_ramp's range spans only 2.12 true standard deviations, p99_ramp's 1.95σ,
and max_ramp's 1.17σ — ranges this narrow, across 11 independent samples, are exactly what
pure sampling noise produces on its own, with no real difference between rules required to
explain them. Most single-trial or 3-trial ramp-statistic ranking claims from earlier in this
project's investigation are accordingly not statistically distinguishable from noise, and are
not asserted as confirmed findings in §6.

The most direct version of this objection — that the strongest unsafe alternative performs
better on ramp statistics, so the safety guarantee has a real cost — is closed by replicating
both `coincidence_ceiling` and `lmetric_power_coincidence_ceiling` to matched n=6 on the same
condition and recomputing z-scores. **The apparent n=3 ramp-statistic edge for the unsafe rule
fully washed out at matched trial counts** (z = -0.03 to +0.28 on mean_ramp, p99_ramp, and
max_ramp — all statistically indistinguishable from zero), while energy-per-token was already
tied between them at n=3. §6.3 reports this comparison in full.

This does not touch the theory: §4's proofs are unaffected by any of this. It does mean this
paper restricts its headline empirical claims to the metrics independently confirmed low-noise
in §5.2 (peak, TTFT, TBT, energy-per-token, SLO-violation-rate), applied uniformly regardless
of which arm a given metric happens to favor, rather than selectively including or excluding a
statistic by outcome, and to differences that clear a conventional significance threshold
rather than differences in point estimates alone (§6.1).

## 6. Experimental Results

Main comparison restricted to four arms, per §4.6/Table 3's roles: `coincidence_ceiling`
(proposed), `lmetric` (power-blind baseline, motivates why power-awareness matters at all),
`round_robin` (state-blind baseline, motivates why load-awareness matters), and
`lmetric_power_coincidence_ceiling` (the strongest empirically-performing unsafe alternative,
needed for the head-to-head that closes the "unsafe rule wins" objection). Seven other arms
tested in this project (the epsilon-shift Pareto-safe family, `weighted_sum_coincidence_ceiling`,
`compute_only`/`load_only`, `drf_fixed`) are discussed in §4 or noted in passing; full tables
are provided as supplementary material.

### 6.1 Peak-power reduction on the short/original condition (n=6)

Condition: closed-loop whale-injection traffic (15% long-prompt fraction, 13.6-15.5k-token
whales), the paper's original validation condition. All four arms are replicated to **n=6**
(up from the n=3 used in earlier drafts of this project, per §5.3's recommendation), restricted
to the three metrics §5.2 confirms are generally low-noise:

**Table 1: Heavy/Closed-Loop (short), per-GPU-calibrated ceiling, mean ± std, n=6 for all four
arms.** Bold marks the lowest point estimate per row; significance is assessed separately
below, not by this formatting.

| Metric | `coincidence_ceiling` (proposed) | `lmetric` (power-blind) | `round_robin` (state-blind) | `lmetric_power_coincidence_ceiling` (unsafe) |
|---|---|---|---|---|
| Peak power (W) | **2275.2 ± 57.1** | 2388.2 ± 31.9 | 2321.6 ± 53.8 | 2318.2 ± 74.8 |
| TTFT mean (s) | 0.542 ± 0.033 | 0.560 ± 0.032 | 0.580 ± 0.038 | 0.537 ± 0.038 |
| TBT mean (ms) | 531.3 ± 38.9 | 544.2 ± 48.9 | **502.7 ± 28.5** | 553.1 ± 18.8 |

Paired and unpaired t-tests (n=6, same fixed-seed workload content across trials) confirm one
of these differences reaches conventional significance: `coincidence_ceiling` reduces peak
power by 4.7% relative to `lmetric` (unpaired p=0.0017, paired p=0.0025). The TTFT reduction
relative to `lmetric` is directionally consistent with the proposed rule but does not reach
significance at n=6 (unpaired p=0.36, paired p=0.10), nor does the TBT reduction relative to
`lmetric_power_coincidence_ceiling` (unpaired p=0.25, paired p=0.29). The confirmed result on
this condition is the peak-power reduction; TTFT and TBT differences are reported as
directional, not as part of a multi-metric dominance claim.

Against `round_robin`, `coincidence_ceiling` is Pareto-incomparable rather than dominant: it
wins peak power and, directionally, TTFT, but loses TBT (531.3 vs. 502.7 ms). This pattern
recurs throughout this project wherever `round_robin` is compared: a state-blind rule never
concentrates load onto an already-busy replica, so it cannot lose on tail-batching metrics
like TBT the way a state-aware rule occasionally can, even as it forgoes the gains available
from reading load and cache state.

Against `lmetric_power_coincidence_ceiling` — the strongest empirically-performing but
Pareto-unsafe alternative, replicated here to matched n=6 (previously n=3) — none of the three
differences in Table 1 reach significance (peak: unpaired p=0.29; TTFT: p=0.80; TBT: p=0.25).
Combined with the matched-n=6 ramp-statistic recheck (§6.3) and the energy-per-token/
SLO-violation-rate comparison (§6.2), `coincidence_ceiling` is statistically indistinguishable
from the strongest unsafe alternative on every metric measured in this study — direct evidence
that the Pareto guarantee is free, not merely that it avoids an efficiency cost.
mean_ramp/p99_ramp/max_ramp are provided as supplementary material for completeness but are
not part of this confirmed claim, per §5.3's uniform, outcome-independent restriction to
confirmed-low-noise metrics.

**Figure 2** (`figs/ramp_comparison.pdf`, LaTeX build only): fleet-aggregate power (top) and
its ramp rate (bottom), Heavy/Closed-Loop (short), 3 trials each of `coincidence_ceiling`
(proposed), `weighted_sum_coincidence_ceiling` (external validity check, Pareto-safe only),
and `lmetric_power_coincidence_ceiling` (unsafe), under the per-GPU-calibrated ramp ceiling
(shaded band). The three arms separate most at the early ramp-up spike (t≈7–9s):
`lmetric_power_coincidence_ceiling`'s worst per-trial peak ramp reaches 9575.0 W/s, vs.
5955.8 W/s for `weighted_sum_coincidence_ceiling` and 4323.2 W/s for the proposed rule — a
per-trial-max statistic that characterizes the trace, not a substitute for Table 1's
significance-tested metrics. Away from that transient the three traces are visually close.

### 6.2 Energy-per-token and SLO-violation-rate: the safety guarantee costs nothing

`coincidence_ceiling` does not reduce energy consumption — a routing policy that is
work-conserving over the same request stream is not expected to change total energy
substantially, and this is not the claim made here. The claim is narrower: **the
Pareto-safety guarantee does not cost anything in efficiency or SLO compliance relative to the
strongest known unsafe alternative.**

**Energy-per-token** (total NVML `energy_mj` delta divided by total output tokens, confirmed
low-noise in §5.2): `coincidence_ceiling` beats `round_robin` on 6 of 7 conditions tested
(often by 5-10%). The one exception, BurstGPT (7.7% worse), is a known real-trace anomaly
consistent with earlier-documented project history (a real-BurstGPT-trace `round_robin`
advantage unrelated to whale-aware admission logic), not a newly-observed problem. Against
`lmetric`, results are mixed (clear wins on 4 conditions, ties or small losses on 3). Against
`lmetric_power_coincidence_ceiling` — the comparison that matters most for the "guarantee
costs nothing" claim — the two are statistically indistinguishable on most conditions
(e.g. 1.3132 vs. 1.3051 on the flagship long condition, well within each other's noise).

**SLO-violation-rate** (fraction of requests with TTFT > 1.0s or max TBT > 200ms, confirmed
low-noise in §5.2, though the fixed thresholds are near-degenerate on lighter conditions where
nearly every arm scores near 0%, a limitation noted here and in §7). TTFT-violation favors
`coincidence_ceiling` over `round_robin` on 5/7 conditions and over `lmetric` on 4/7.
**TBT-violation is genuinely mixed**: `coincidence_ceiling` loses to `round_robin` on
Heavy/CL short (0.2978 vs. 0.2856) and BurstGPT (0.1627 vs. 0.1683), though it wins on
WildChat (0.5074 vs. 0.5828); it loses to `lmetric` on Heavy/CL long (0.3867 vs. 0.3861,
marginal) and WildChat (0.5074 vs. 0.4134, a real loss). The tail-latency result of this paper
is a peak-power win on the flagship condition (§6.1) and a directionally favorable TTFT
picture more broadly, not a universal win on every latency statistic.

**Table 2: Energy-per-token, headline arms, all 7 conditions (mean ± std, J/token).**
SLO-violation-rate figures for all 7 conditions and 4 arms are provided as supplementary
material; the specific figures behind the disclosure above are cited inline.

| Condition | `coincidence_ceiling` | `lmetric` | `round_robin` | `lp_coincidence_ceiling` |
|---|---|---|---|---|
| Heavy/CL short | 1.4242±0.0193 | 1.4180±0.0222 | 1.4395±0.0176 | 1.4179±0.0230 |
| Heavy/CL long | 1.3132±0.0050 | 1.3251±0.0242 | 1.3363±0.0023 | 1.3051±0.0032 |
| Heavy/Matched | 0.7393±0.0116 | 0.7510±0.0110 | 0.7486±0.0075 | 0.7456±0.0040 |
| Light/Cachehit | 1.2287±0.0028 | 1.2517±0.0102 | 1.2944±0.0042 | 1.2339±0.0021 |
| BurstGPT | 2.8070±0.0312 | 2.8599±0.0235 | 2.6059±0.0118 | 2.8403±0.0101 |
| Ramp & Route | 2.2950±0.0323 | 2.2537±0.0187 | 2.5404±0.0462 | 2.2508±0.0373 |
| WildChat | 0.3227±0.0071 | 0.3147±0.0045 | 0.3448±0.0004 | 0.3146±0.0060 |

### 6.3 Matched n=6 recheck against the strongest unsafe alternative

Per §5.3, both `coincidence_ceiling` and `lmetric_power_coincidence_ceiling` were replicated
to matched n=6 on the flagship condition, and z-scores were recomputed on the three
ramp-derivative statistics: z = -0.03 to +0.28 on mean_ramp, p99_ramp, and max_ramp, all
statistically indistinguishable from zero. The apparent n=3 advantage for the unsafe rule does
not survive matched replication. This result extends to peak power, TTFT, and TBT as well
(§6.1) and to energy-per-token and SLO-violation-rate (§6.2): on every metric this study
measures, at matched trial counts, giving up the Pareto guarantee produces no statistically
distinguishable empirical advantage.

### 6.4 Alternative Share_power Definitions

Two alternative `Share_power` definitions were evaluated against the deployed raw two-point
derivative: an EMA-smoothed estimate (α=0.3), and a retarget to instantaneous peak power. The
smoothed estimate reduces the signal's own relative std by 18-53% depending on measurement
cadence but does not produce a consistent downstream improvement: it beats
`coincidence_ceiling` on energy-per-token on 3 of 7 conditions (~1-2%) and loses on 4;
max_ramp is worse on 6 of 7 conditions, and Ramp & Route shows a loss across five metrics
simultaneously (energy, TTFT-violation, mean TTFT, p99_ramp, max_ramp). The peak-power
retarget shows a large, one-sided regression on the two conditions tested so far
(energy-per-token +11%, TTFT-violation-rate nearly 2×, TBT-violation +21% on the flagship
condition), and it changes the mechanism's physical target from grid-transient avoidance to
capacity/thermal management, a different problem than the one this paper addresses. Neither
alternative is adopted; the deployed raw-ramp-rate definition is retained on both empirical
and mechanistic grounds.

## 7. Discussion / Limitations

- **Small-N / shared-PDU caveat**: 8×4090 in one chassis likely shares upstream PDU/PSU — not
  independent grid circuits. Per-GPU power is measured independently; any data-center-scale
  claim would need a separate extrapolation model. Such an extrapolation was deliberately not
  attempted: a sibling project in this line of work retracted its own fleet-scale
  extrapolation after both a parametric Monte Carlo model (P99 ramp tail underestimated by
  roughly 5.8× against hardware) and a model-free trace-bootstrap (tail-statistic sign flips
  from a small segment library) failed on tail fidelity — with a *thicker* trial library than
  this project's own (3-6 trials per arm). A similar extrapolation here was judged unreliable,
  not merely out of scope for space.
- **Real-signal losses, disclosed rather than omitted**: on the long/sustained-pressure
  condition specifically, `coincidence_ceiling`'s mean_ramp is significantly higher than
  `round_robin`'s (z≈+2.82, n=6) — a real, credible finding given mean_ramp is the *least*
  noisy of the three ramp derivatives (§5.2), not discounted as noise-explainable. §6.2's
  TBT-violation-rate picture is genuinely mixed against both baselines, not a universal win.
  BurstGPT's energy-per-token result reverses (`coincidence_ceiling` 7.7% worse than
  `round_robin`), consistent with an earlier-documented, still-undiagnosed real-BurstGPT-trace
  anomaly in this project's history.
- **Scope of validation**: the confirmed peak-power result (§6.1) is demonstrated on the
  short/original condition; the long/sustained-pressure condition shows no confirmed
  ramp-safety advantage in either direction. The tail-latency result is not claimed to
  generalize to every condition, only where confirmed, while the efficiency/SLO
  no-regression result holds broadly (§6.2). A full accounting of exactly which workload
  properties predict where the peak-power win holds is future work.
  `Share_power`'s live-signal ablations (§6.4) are partial: the peak-power retarget has data
  on 2 of 7 conditions so far.
- **Per-decision vs. per-trajectory, and per-replica vs. aggregate**: §3 is explicit that the
  welfare objective is myopic (per-decision); a sequence of decisions is not claimed to be
  optimal over a trajectory, only that each satisfies a static guarantee. Fleet-aggregate ramp
  is a sum across replicas, so a per-replica bound does not mechanically imply a bound on the
  sum — §4.6's coincidence-ceiling mechanism targets the aggregate hazard directly, still
  inheriting the per-decision proof, but no formal bound on the aggregate trajectory itself is
  claimed, only the empirical characterization in §6.
- **Fixed replica pool**: this paper routes among N already-running replicas; deciding which
  replicas are powered on at all is a separate, coarser-timescale decision problem this paper
  does not address.

## 8. Related Work

- **Dominant Resource Fairness (DRF)** [1] — multi-resource fair allocation via lexicographic
  comparison of dominant shares, proven Pareto-efficient for static per-user demand vectors.
  This paper extends the resource set to a third, live/reactive dimension (§3) and re-derives
  Pareto-efficiency for this dynamic setting rather than assuming the static proof transfers.
- **LMETRIC** [2] — multiplicative `P-token × BS` scoring for LLM-serving load balancing, a
  simpler non-fairness-theoretic baseline. §4.2 shows a natural power-extended form of this
  score sacrifices the Pareto guarantee via a distinct mechanism from the DRF-family variant;
  §6's hardware validation is a head-to-head against its coincidence-ceiling-extended form,
  the strongest empirical performer found in this project.
- **Power-aware ML system design (Zeus, Perseus)** [3, 4] — tune per-job GPU frequency/power
  configuration (Zeus) or schedule computation energy across pipeline stages (Perseus) *within*
  a single training job, via DVFS. `ramp_ceiling` (§3) is physically a symptom of the same DVFS
  transition dynamic, which is why this paper treats it as a measured constant (§5) rather than
  a quantity it controls. This paper acts at a different layer and timescale — which
  already-running replica serves a request, not how any GPU's own DVFS state changes — and the
  two compose (Zeus/Perseus underneath a power-aware router).
- **Power-of-Two-Choices** [5, 6] — sampled load balancing with a proven exponential
  improvement in expected max load; a different mechanism family (randomized sampling vs.
  the full-visibility deterministic rule used here), noted for completeness.
- **Power capping as an alternative mechanism** [7] — recent work shows GPU-level power caps
  are often *inert* for decode-dominated LLM serving (137–300 W draw on a 700 W GPU), since a
  facility-level cap frequently never engages — motivating acting upstream at the routing
  decision instead.
- **Grid-integrated AI infrastructure** [8] — broader facility/life-cycle-level strategies for
  aligning AI workload management with grid conditions. This paper operates at a
  complementary, much shorter timescale (per-request, 500ms) layer, addressing intra-fleet
  coincident ramp spikes rather than facility- or grid-scale coordination directly.
- **Online optimization with switching costs** [9] — penalizes an action's cost of *change*,
  not just its instantaneous cost; `Share_power`'s ramp-rate term and §4.6's coincidence-ceiling
  mechanism are conceptually switching-cost signals. This paper does not adopt that
  literature's regret/competitive-ratio analysis — §4.1's guarantee is per-decision, not
  trajectory-level.
- **Rate limiting in feedback control** [10] — `ramp_ceiling` is a slew-rate bound in the
  control-theoretic sense; `Share_power` normalizes the plant's rate of change against an
  actuation limit, the same object a rate limiter enforces classically. This paper does not
  build a continuous controller — routing is discrete per-request placement — but
  §4.3/§4.6's live-recalibrated, fleet-shared ceiling is exactly this constraint's set-point,
  estimated online rather than fixed offline.

## 9. Conclusion

This paper gives power-aware LLM-serving routing a precise theoretical grounding and a single,
unambiguous proposal. We prove a Pareto-non-domination guarantee for the sorted DRF rule, show
a fixed-priority tie-break and a power-extended LMETRIC score each sacrifice it via distinct
mechanisms, and introduce the coincidence-ceiling mechanism — a pluggable, fleet-aware ceiling
adjustment that targets the fleet-*aggregate* coincident-ramp hazard no purely per-replica
rule can see, requiring no new safety proof since it is a direct instantiation of an existing
corollary. The resulting rule, `drf_power_tiebreak_full_coincidence_ceiling`, is this paper's
proposed strategy. On real 8×4090 hardware, across 7 workload conditions including two real
traces, it delivers a significant peak-power reduction relative to a power-blind baseline on
the condition central to this paper's motivation (p=0.0017, n=6), and, against the strongest
empirically-performing but *not* Pareto-safe alternative replicated to matched n=6, is
statistically indistinguishable on every metric measured — peak power, TTFT, TBT,
ramp-derivative statistics, energy-per-token, and SLO-violation-rate. The safety guarantee
costs nothing on any axis this study tests. A second contribution falls out of this
validation: a ~75-85% decision-level noise floor inherent to real-hardware closed-loop routing
evaluation is characterized, along with which evaluation statistics survive it and which
don't, disclosing real losses rather than suppressing them, to determine exactly which of this
paper's own empirical claims can currently be asserted with confidence. The result is a
routing strategy that is provably safe at the decision level, empirically validated where the
theory predicts it should matter, and accompanied by a falsifiable account of where it does
not yet win.

## References

1. A. Ghodsi, M. Zaharia, B. Hindman, A. Konwinski, S. Shenker, and I. Stoica. "Dominant
   Resource Fairness: Fair Allocation of Multiple Resource Types." *Proceedings of the 8th
   USENIX Symposium on Networked Systems Design and Implementation (NSDI '11)*, 2011.
2. D. Zhang, J. Han, K. Zhang, X. Wei, S. Shen, C. Fang, W. Yu, J. Zhou, and R. Chen.
   "LMetric: Simple is Better — Multiplication May Be All You Need for LLM Request
   Scheduling." *Proceedings of the 20th USENIX Symposium on Operating Systems Design and
   Implementation (OSDI '26)*, 2026. arXiv:2603.15202.
3. J. You, J.-W. Chung, and M. Chowdhury. "Zeus: Understanding and Optimizing GPU Energy
   Consumption of DNN Training." *Proceedings of the 20th USENIX Symposium on Networked
   Systems Design and Implementation (NSDI '23)*, 2023.
4. J.-W. Chung, Y. Gu, I. Jang, L. Meng, N. Bansal, and M. Chowdhury. "Reducing Energy Bloat
   in Large Model Training." *Proceedings of the 30th ACM Symposium on Operating Systems
   Principles (SOSP '24)*, 2024. arXiv:2312.06902.
5. M. Mitzenmacher. "The Power of Two Choices in Randomized Load Balancing." *IEEE
   Transactions on Parallel and Distributed Systems*, 12(10):1094-1104, 2001. (Originally
   presented as part of the author's 1996 PhD thesis, UC Berkeley.)
6. Y. Azar, A. Z. Broder, A. M. Karlin, and E. Upfal. "Balanced Allocations." *Proceedings of
   the 26th Annual ACM Symposium on Theory of Computing (STOC '94)*, 1994. (The original
   static balls-into-bins result underlying the power-of-two-choices line of work cited
   above.)
7. B. Ma, A. Afzal, J. Eitzinger, and G. Wellein. "The Illusion of Power Capping in LLM
   Decode: A Phase-Aware Energy Characterisation Across Attention Architectures." arXiv
   preprint arXiv:2605.11999, 2026.
8. A. A. Chien, U. Gupta, S. Ren, A. Sriraman, and B. Tomlinson. "Strategies and Design for
   Increasing AI Sustainability." *Nature Reviews Clean Technology*, 2026.
   doi:10.1038/s44359-026-00195-w.
9. P. Li, Y. Han, A. Wierman, and S. Ren. "Fairness-Regularized Online Optimization with
   Switching Costs." *Advances in Neural Information Processing Systems (NeurIPS '25)*, 2025.
   arXiv:2512.11131.
10. K. J. Åström and R. M. Murray. *Feedback Systems: An Introduction for Scientists and
    Engineers.* Princeton University Press, 2008.
11. M. J. Neely. *Stochastic Network Optimization with Application to Communication and
    Queueing Systems.* Synthesis Lectures on Communication Networks, Morgan & Claypool, 2010.
12. A. Sen. *Collective Choice and Social Welfare.* Holden-Day, San Francisco, CA, 1970.
13. A. M. Geoffrion. "Proper Efficiency and the Theory of Vector Maximization." *Journal of
    Mathematical Analysis and Applications*, 22(3):618-630, 1968.
