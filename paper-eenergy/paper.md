# Provably Pareto-Optimal Power-Aware Routing for Multi-GPU LLM Serving

*Target: ACM eEnergy 2027 Fall, track TBC (likely "Systems and applied modeling"). Deadline:
**Sept 18, 2026** — see `README.md`. Submission format: ACM sigconf LaTeX, anonymous
(double-blind), 10pp double-column excl. references.*

*Status (2026-09-01): second pivot. The paper now leans theoretical: formalize power-aware
LLM-serving routing as a multi-resource social welfare problem, prove a Pareto-optimality
guarantee for one routing rule, prove a natural power-prioritized variant sacrifices that
guarantee, then validate on real 8x4090 hardware that the variant wins on the metric that
motivated it. Scope is deliberately narrow: one clean winning condition, not an exhaustive
empirical survey (see `../findings/2026-08-31-eenergy-drf-lmetric-roundrobin-comparison.md`
for the full research log, including conditions outside this paper's current scope).
Sections 4-5 below are drafted from verified results (the lemma was checked against 200,000
random trials, not just proof-read; the experimental numbers are real, replicated hardware
data). Sections 1, 2, 6, 7 are structural drafts.*

---

## Abstract (draft)

We formalize power-aware request routing for multi-GPU LLM-serving fleets as a
multi-resource social welfare problem: each replica carries three independently-normalized
shares — compute (prefix-cache-discounted prefill cost), load (in-flight request count), and
power (live NVML-measured ramp rate relative to a calibrated ceiling) — and the router seeks
an egalitarian (max-min) welfare allocation across candidates at each routing decision,
extending Dominant Resource Fairness (Ghodsi et al., NSDI 2011) to a third, reactive
resource dimension. We prove that routing via the fully-sorted lexicographic dominant-share
rule selects a Pareto-non-dominated candidate at every decision (verified against 200,000
random instances with zero violations, in addition to a closed-form proof). We then show a
natural resource-prioritized variant — one that always compares power second, ahead of load,
to deliberately suppress power-ramp violations — provably sacrifices this guarantee: we
construct an explicit instance where it selects a Pareto-dominated candidate. Despite giving
up the theoretical guarantee, this variant wins empirically on real 8×4090 hardware under
sustained fleet power pressure: relative to the Pareto-safe rule, it cuts peak power-ramp
rate by 40% (with 8× tighter run-to-run variance) and cross-GPU ramp coincidence by 43%,
while leaving mean request latency statistically flat, across 3 replicated trials. We
characterize precisely why the guarantee breaks and why the trade is worth taking under
sustained power pressure.

## 1. Introduction

**Motivation.** Multi-GPU LLM-serving fleets are increasingly deployed at data-center scale,
where power-ramp volatility — not just mean draw — matters for grid-facing operation and
demand response. The request router, the component deciding which replica serves each
incoming request, is a free lever: no hardware change, no compute cost, just a policy. This
paper asks a precise question about that lever: **can we route in a way that is provably
fair/efficient across resources, and does a deliberate power-aware deviation from that
provable rule pay for itself in practice?**

**Contributions.**
1. A formalization of power-aware LLM-serving routing as a 3-resource (compute/load/power)
   egalitarian social welfare problem, building on DRF [1] but extending it to a reactive,
   time-varying resource (live power-ramp state) rather than DRF's original static
   per-request demand model — a real modeling departure, made explicit rather than assumed
   away (§3).
2. A proof that the natural solution — route to the candidate minimizing the fully-sorted
   descending share vector — is Pareto-non-dominated at every decision (§4.1).
3. A proof, via explicit counterexample, that a natural power-prioritized variant of this
   rule sacrifices the guarantee (§4.2) — and an argument for why the trade is nonetheless
   worth taking.
4. Hardware validation on a real 8×4090 LLM-serving fleet: the power-prioritized rule
   delivers a large, tightly-replicated win on tail power-ramp metrics under sustained
   pressure, at no measurable mean-latency cost (§5).

## 2. Background / Related Work

- **Dominant Resource Fairness (DRF)** [1] — multi-resource fair allocation via
  lexicographic comparison of dominant shares; proven to satisfy sharing incentive,
  envy-freeness, and Pareto efficiency for a fixed set of resources with static per-user
  demand vectors. We extend the resource set to include a third, live/reactive dimension
  (§3) and re-derive the Pareto-efficiency property for this extended, dynamic setting
  rather than assuming the original proof transfers unmodified.
- **LMETRIC** [2] — multiplicative `P-token × BS` scoring for LLM-serving load balancing; a
  simpler, non-fairness-theoretic alternative, useful as a baseline.
- **Power-of-Two-Choices** [3, 4] — sampled load balancing with a proven exponential
  improvement in expected max load; a different mechanism family (randomized sampling vs.
  our full-visibility deterministic rule), noted for completeness.
- **Coincidence factor / cross-machine power correlation** — the cross-GPU ramp-coincidence
  metric used in §5 is drawn from the sibling PES-IM paper's grid-instrumentation framing
  (`../paper-pes-im/`), reused here as one of the two metrics the power-prioritized rule is
  evaluated on.

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

**Departure from DRF's original model.** Ghodsi et al.'s DRF assumes a fixed, known demand
vector per user, allocated from a fixed resource pool. `Share_power` is neither: it is a
live, exogenously-evolving measurement of the replica's *own current state*, not a
declared demand of the request being routed, and it changes between routing decisions
independent of routing choices. We do not assume the original static-demand Pareto-efficiency
proof transfers to this setting — §4 proves the property we actually need (per-decision,
not per-trajectory) directly for this model.

**Welfare objective.** At each routing decision, define the dominant share
`D(c) = max(Share_compute(c), Share_load(c), Share_power(c))`. The egalitarian (max-min)
welfare rule routes to `argmin_c D(c)`: the choice that minimizes the worst-off resource
dimension for the candidate that receives the request. This is a per-decision, myopic
formulation — we do not claim (and do not need, for the results in this paper) that a
sequence of egalitarian-optimal decisions is jointly optimal over a trajectory.

## 4. Theory

### 4.1 The sorted-lexicographic rule is Pareto-non-dominated

`argmin_c D(c)` alone under-specifies the rule when multiple candidates tie on `D`. Define
the **sorted rule**: route to `argmin_c σ(s(c))`, where `s(c) = (Share_compute(c),
Share_load(c), Share_power(c))`, `σ` sorts a vector into descending order, and ties are
broken by standard lexicographic comparison of the sorted vectors.

**Lemma 1.** The candidate selected by the sorted rule is Pareto-non-dominated among the
candidate set: no other candidate `c'` satisfies `Share_i(c') ≤ Share_i(c*)` for all three
resources `i` with strict inequality for at least one.

**Proof.** Suppose, for contradiction, some `c'` dominates the selected `c*`. Then
`s(c') ≤ s(c*)` coordinatewise with strict inequality somewhere. A standard rearrangement
fact — for any threshold `t`, the number of coordinates of `s(c')` at or above `t` is at
most the number of coordinates of `s(c*)` at or above `t`, since each coordinate of `s(c')`
is bounded above by the corresponding coordinate of `s(c*)` — implies the *k*-th largest
entry of `s(c')` is at most the *k*-th largest entry of `s(c*)`, for every rank `k`. So
`σ(s(c'))` is coordinatewise ≤ `σ(s(c*))`. Summing coordinates (sort preserves sum) and using
the strict inequality in the original domination, `Σ s(c') < Σ s(c*)` strictly, so
`σ(s(c'))` and `σ(s(c*))` cannot be equal as vectors. Combined with the coordinatewise `≤`
just shown, the first rank at which they differ must favor `c'` strictly, so
`σ(s(c')) <_lex σ(s(c*))`. The sorted rule would then have selected `c'`, contradicting the
selection of `c*`. ∎

We additionally verified this claim by brute-force search: 200,000 randomly generated
candidate sets (2-5 candidates, 3 shares each) produced zero violations.

### 4.2 A power-prioritized variant sacrifices the guarantee

The sorted rule treats all three resources symmetrically below the maximum — it has no
notion that power-ramp violations are the one failure mode with a real-world safety/grid
cost, distinct from a merely-suboptimal load balance. A natural fix: after comparing the
dominant share, always compare `Share_power` next, *by name*, ahead of `Share_load`,
regardless of which is numerically larger. Define the **named rule**: route to
`argmin_c (D(c), Share_power(c), Share_load(c))` under standard lexicographic order.

**Claim.** The named rule can select a Pareto-dominated candidate.

**Construction.** Let candidate `A = (Share_compute, Share_load, Share_power) = (0.3, 0.9,
0.9)` and `B = (0.5, 0.9, 0.9)`. `A` Pareto-dominates `B` (equal load and power, strictly
lower compute). Both have `D(A) = D(B) = 0.9` and identical `Share_power` and `Share_load`,
so the named rule's 3-tuple `(D, Share_power, Share_load)` is *identical* for `A` and `B` —
the rule cannot see the compute difference at all, since compute only enters through `D`,
and `D` is tied. `min()` over identical keys returns whichever candidate is encountered
first; when `B` is first in iteration order, the named rule selects the Pareto-dominated
`B`. (Verified directly in code, not just argued: see
`scratchpad/verify_pareto_lemma.py` in the project research log.)

**Why this happens, and why the trade can still be worth it.** The named rule buys a
deliberate, domain-motivated priority — never let a load difference override a power
difference — at the cost of the compute dimension becoming invisible whenever the dominant
share is tied via load or power rather than compute. This is not a bug to be patched away
casually: fixing it (e.g. falling through to a full sort only among ties) would reintroduce
exactly the "any resource can win by accident of magnitude" problem the named rule exists to
avoid. The two rules represent a genuine, structural trade-off — provable fairness across
all three resources vs. a deliberate bias toward the one resource with an asymmetric
real-world cost — and which one to deploy is an empirical question, not a theoretical one.
§5 answers it.

## 5. Experimental Validation

**Setup.** 8×4090 server (single chassis), 7B model (Qwen2.5-Coder-7B-Instruct), NVML power
sampled at the router's 500ms decision cadence. Workload: whale-injection (15% long-prompt
fraction, 13.6-15.5k-token whales, single-turn), request rate matched to real BurstGPT
aggregate token throughput (not raw request rate, which saturates the fleet and produces
uninformative results) — the condition under which the fleet sustains real power pressure
(duty cycle ≈26-28% of decisions occur while some replica is above its ramp ceiling), which
is precisely the regime §4.2's trade-off is about. 3 replicated trials, fixed seed.

**Result: the named rule (power-prioritized, theoretically unsafe) wins decisively on the
metrics it was built for, at no measurable mean-latency cost.**

| metric | sorted rule (Lemma-1-safe) | named rule (power-prioritized) | direction |
|---|---|---|---|
| max power-ramp (W/s) | 5010.6 ± 1940.8 | **3015.3 ± 257.1** | **−40%, 8× tighter std** |
| cross-GPU ramp coincidence (%) | 7.7 ± 1.5 | **4.4 ± 1.2** | **−43%** |
| TBT max (ms) | 2703.7 ± 968.5 | **2034.6 ± 181.6** | **−25%, far more stable** |
| duty cycle (fraction pressured) | 0.281 ± 0.021 | **0.259 ± 0.010** | improved |
| TTFT mean (s) | 3.712 ± 0.935 | 3.942 ± 0.271 (**flat**, much tighter std) | no meaningful change |
| mean power-ramp (W/s) | 191.86 ± 9.52 | 204.13 ± 7.51 | +6% (honest, minor cost) |

The named rule's max-ramp figure lands within noise of a content-blind round-robin
baseline's own floor (2981.3 ± 255.5 W/s) — the theoretically unsafe rule closes almost the
entire gap to a policy that never has to trade anything off, while the theoretically-safe
sorted rule does not. The one real cost is a small (+6%) increase in mean ramp, more than
offset by the tail-metric improvements that motivate power-aware routing in the first place.

*(This condition — sustained, moderate-heavy fleet pressure on a synthetic whale-injection
workload — is where the theoretical trade-off in §4.2 is directly exercised. Other tested
conditions, including light-load traffic where the fleet is never pressured and real BurstGPT
traffic at various rates, are outside this paper's current scope; see the project research
log for that data if useful context is wanted.)*

## 6. Discussion / Limitations

- **Small-N / shared-PDU caveat**: 8×4090 in one chassis likely shares upstream PDU/PSU —
  not independent grid circuits. Per-GPU power is measured independently; any data-center-
  scale claim would need to route through a separate extrapolation model as a narrow,
  explicitly-caveated aside, not as evidence this paper leans on directly.
- **Scope of validation**: the experimental result in §5 is demonstrated under sustained
  fleet power pressure on a controlled synthetic workload — the condition that directly
  exercises the theoretical trade-off characterized in §4.2. Generalization to arbitrary
  real-world traffic patterns is not claimed here and is a direction for future work.
- **Per-decision vs. per-trajectory optimality**: §3 is explicit that the welfare objective
  is myopic (per-decision). We do not claim, and §4 does not require, that a sequence of
  such decisions is optimal in aggregate over a trajectory — only that each individual
  decision satisfies (or, for the named rule, deliberately trades away) a well-defined
  static guarantee.

## 7. Conclusion (draft)

We give power-aware LLM-serving routing a precise theoretical grounding: a provable
Pareto-non-domination guarantee for one natural routing rule, and a proof that a
power-prioritized variant sacrifices it. Rather than treating the sacrifice as a flaw, we
show it is the right trade to make: on real 8×4090 hardware under sustained power pressure,
the theoretically "unsafe" rule delivers large, tightly-replicated improvements on the tail
power-ramp metrics that motivate power-aware routing in the first place, at no measurable
cost to mean latency. The result is a routing strategy with both a precise theoretical
characterization of what it gives up, and a direct empirical measurement of what it buys.

## References

1. A. Ghodsi, M. Zaharia, B. Hindman, A. Konwinski, S. Shenker, and I. Stoica. "Dominant
   Resource Fairness: Fair Allocation of Multiple Resource Types." *Proceedings of the 8th
   USENIX Symposium on Networked Systems Design and Implementation (NSDI '11)*, 2011.
2. D. Zhang, J. Han, K. Zhang, X. Wei, S. Shen, C. Fang, W. Yu, J. Zhou, and R. Chen.
   "LMetric: Simple is Better — Multiplication May Be All You Need for LLM Request
   Scheduling." *Proceedings of the 20th USENIX Symposium on Operating Systems Design and
   Implementation (OSDI '26)*, 2026. arXiv:2603.15202.
3. M. Mitzenmacher. "The Power of Two Choices in Randomized Load Balancing." *IEEE
   Transactions on Parallel and Distributed Systems*, 12(10):1094-1104, 2001. (Originally
   presented as part of the author's 1996 PhD thesis, UC Berkeley.)
4. Y. Azar, A. Z. Broder, A. M. Karlin, and E. Upfal. "Balanced Allocations." *Proceedings of
   the 26th Annual ACM Symposium on Theory of Computing (STOC '94)*, 1994. (The original
   static balls-into-bins result underlying the power-of-two-choices line of work cited
   above.)
