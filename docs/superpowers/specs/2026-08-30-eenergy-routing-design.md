# DRF-based power-aware routing for ACM e-Energy 2027

*Status: design agreed 2026-08-30, nothing built yet. Companion to `paper-eenergy/README.md`
and `project_eenergy_routing_paper.md` (memory). Target venue: ACM e-Energy 2027 (deadline
unconfirmed — see open items).*

---

## 1. Problem motivation and literature review

AI data-center electricity demand is a fast-growing, increasingly binding constraint on new
AI infrastructure — global data-center consumption was ~460 TWh in 2022, projected to
~1,050 TWh by 2026, with power capacity now the dominant deployment bottleneck, more than
chip supply ([Electricity Demand and Grid Impacts of AI Data Centers, arXiv:2509.07218](https://arxiv.org/html/2509.07218v1)).
A growing "data centers as flexible grid participants" literature argues AI infrastructure
should actively help the grid — shifting load temporally or geographically, providing fast
frequency response, absorbing renewable surplus
([Power-Flexible AI Data Centers, arXiv:2606.25098](https://arxiv.org/pdf/2606.25098);
[To Defer or To Shift?, arXiv:2604.05376](https://arxiv.org/pdf/2604.05376)).

The specific problem this project targets — power *correlation*, not just power *level* —
has direct precedent on the power-systems side. A 2025 NAPS paper on probabilistic load
coincidence factors and a 2026 paper on
["Spatial Load Correlation in AI Data-Center-Dominated Power Systems" (arXiv:2606.13853)](https://arxiv.org/pdf/2606.13853)
both make the point this project's own coincidence-factor (CF) theory (see sibling
`paper-pes-im/`) independently arrived at: traditional power-provisioning diversity factors
assume individual loads fluctuate independently, but AI clusters violate that — diversity
factor approaches 1.0 (near-synchronous peaks) for active training/inference clusters,
versus 0.6–0.75 for ordinary mixed IT load. That gap between assumed and actual correlation
is what forces operators into either wasteful over-provisioning or breaker-trip/derating
risk.

On the computing side, power-aware GPU cluster scheduling is an active but
differently-aimed literature: frequency-aware dispatchers that route requests to whichever
GPU's current clock state best matches the job
([Festina / Energy-Aware Scheduling for Serverless LLM Serving, arXiv:2606.30391](https://arxiv.org/html/2606.30391v1)),
DVFS-based SLO-aware frequency scaling
([GreenLLM, arXiv:2508.16449](https://arxiv.org/pdf/2508.16449)), and fragmentation-aware
GPU datacenter schedulers. These optimize *energy-per-request* or *SLO-at-lower-frequency*
— none target cross-GPU power *correlation* as the objective. Carbon-aware spatiotemporal
scheduling is the closest thing to a "shape the fleet's aggregate power profile"
literature, but it operates by shifting load across time or geography
([Let's Wait Awhile, arXiv:2110.13234](https://arxiv.org/pdf/2110.13234);
[Carbon-Aware Spatiotemporal Scheduling of Data Transfers](https://dl.acm.org/doi/10.1145/3797248.3815406))
— mechanisms unavailable to a single-location, latency-serving fleet like this project's
8×4090 testbed.

**The specific routing anchor:** the current state of the art for production LLM-serving
routing is **LMETRIC** ("Simple is Better: Multiplication May Be All You Need for LLM
Request Scheduling", to appear OSDI'26,
[arXiv:2603.15202](https://arxiv.org/abs/2603.15202)). It scores each candidate replica as
`P-token × BS` (new prefill tokens needed accounting for KV$ hits, times current batch
size) and routes to the minimum — a multiplicative form chosen specifically to avoid
per-workload hyperparameter tuning. Already deployed in production (Alibaba), reporting
92%/39% TTFT and 24%/51% TPOT improvements over vLLM-v1 and an in-production scheduler on
real traces. **Confirmed from the paper: no power/energy signal is considered anywhere in
its design.** That is the gap this project fills — not "routing can help power" in the
abstract (routing-for-cache-affinity-and-load already exists and is state of the art), but
"can a production-grade router be extended with a power dimension without regressing what
it already guarantees."

**Design-space dead ends already ruled out during brainstorming (kept here so we don't
re-derive them):**
- A raw third multiplicative factor (`P-token × BS × PowerFactor`) has no verified
  interaction guarantee — LMETRIC's own two-term product isn't a literal instance of a
  provable theorem either (its "cancels out" property is scale-invariance to unit choice,
  not proven optimality), so stacking a third factor the same way doesn't inherit anything
  beyond that scale-invariance, and even that breaks once any term needs a *weight*.
- Lyapunov drift-plus-penalty (virtual-queue backpressure) gives a real `[O(1/V), O(V)]`
  trade-off guarantee, but requires a tunable weight `V` — directly contradicting the
  hyperparameter-free design goal that makes LMETRIC citable in the first place.
- Proportional fairness / Nash-bargaining-style products only carry their real
  convergence-to-optimum guarantee (Kushner & Whiting) when *every* term is an
  "instantaneous marginal ÷ own EWMA average" ratio — LMETRIC's `BS` term isn't that
  structure, and the guarantee does not survive a naive 3-term generalization.

## 2. Problem statement

Data-center power infrastructure is provisioned using a diversity/coincidence factor that
assumes not all connected load peaks simultaneously. AI clusters violate that assumption
(diversity factor → 1.0), forcing operators into over-provisioning (wasted capital) or
under-provisioning (breaker-trip/derating risk). Production routing today (LMETRIC)
optimizes purely for cache-affinity and load-balance, with **zero power awareness**,
confirmed directly from the paper.

**The question this project answers:** can a power-aware extension of a production-proven
router reduce cross-GPU power correlation/ramp rate, at a *quantified, honestly-scoped*
latency cost — without requiring any hyperparameter to trade the two off, and without
discarding the cache-affinity/load-balance guarantees the base router already provides?

This is deliberately **not** framed as "our router is strictly better." Adding a power
dimension to a routing decision necessarily costs something on the objectives that used to
have the router's undivided attention — the contribution is in *how little* it costs and
*where* that cost falls, not in denying it exists. See §3.2.

## 3. Routing strategy

### 3.1 Mechanism: Dominant Resource Fairness (DRF) over three resource shares

Ghodsi et al., NSDI 2011
([paper](https://amplab.cs.berkeley.edu/wp-content/uploads/2011/06/Dominant-Resource-Fairness-Fair-Allocation-of-Multiple-Resource-Types.pdf))
— the allocation mechanism behind Mesos. DRF generalizes max-min fairness to multiple,
heterogeneous resource types: each user's **dominant resource** is whichever resource type
takes the largest proportional share of the pool given their consumption pattern; DRF
equalizes dominant *shares* across users rather than raw resource amounts. Proven
properties (any number of resource types, not just two): Pareto efficiency, sharing
incentive, envy-freeness, strategy-proofness, and reduction to ordinary max-min fairness in
the single-resource case.

**Why DRF over the ruled-out alternatives:** its guarantees don't depend on dimension
count — the exact property that broke for the multiplicative/Lyapunov/PF approaches above
when a third term was added.

**Adaptation to this problem — three independently-normalized shares, no cross-resource
weight:**

- **Share_compute,i** = (new prefill tokens needed if routed to replica *i*, accounting for
  KV$ hits) / `token_budget` (replica *i*'s configured max-scheduled-tokens ceiling — a
  real vLLM scheduler config value, not an invented denominator).
- **Share_load,i** = (running + queued requests at *i*, after hypothetically adding this
  request) / `max_num_seqs` (replica *i*'s configured concurrency ceiling).
- **Share_power,i** = (recent or projected power ramp rate at *i*) / ramp-rate ceiling
  (target anchored to the PES-IM gate experiment's measured figures, or set independently —
  open item, see §5).

`D_i = max(Share_compute,i, Share_load,i, Share_power,i)`. Route the incoming request to
whichever replica minimizes the *resulting* `D_i` (compute the what-if `D_i` for each
candidate as if the request were placed there; pick the minimum). No weight between
dimensions anywhere — each share is normalized to its own real capacity, so no
cross-resource unit-conversion hyperparameter is needed either.

**Implementation components** (all in the router, no new vLLM hotpatches required beyond
what already exists for telemetry):
- A per-replica prefix/radix-tree mirror of cached blocks, updated as the router dispatches
  requests, to compute `Share_compute,i` without querying the replica directly (mirrors how
  LMETRIC and prefix-aware routers generally avoid a round-trip per decision).
- Live running+queued counters per replica, updated from request lifecycle events the
  router already observes (dispatch → completion) — no polling needed for `Share_load,i`.
- Reuse of `scripts/pesim/power_logger.py`'s NVML sampling (already multi-GPU capable, no
  changes needed) for `Share_power,i`.

**Honest scope caveat:** DRF's original theorem is proven for a static multi-tenant
allocation game (dividing a fixed resource pool fairly among users), not a literal online
streaming routing decision. Applying it here adapts the *principle* (equalize normalized
dominant share, no cross-resource weight needed) to a greedy per-request setting — the same
kind of honest gap LMETRIC itself has relative to true optimality; the paper should not
claim a literal DRF-theorem guarantee for the online routing setting, only motivate the
mechanism by it and validate empirically.

### 3.2 What this optimizes for — claim framing

DRF is a **fairness** mechanism, not a latency-minimizer. Giving power equal standing
alongside compute and load necessarily means some requests get routed sub-optimally on
compute/load *when power is the binding constraint* for a candidate replica — this is the
mechanism working as intended, not a flaw to explain away.

**Claims this design will and will not make:**
- Will **not** claim Pareto-dominance over a latency-only router (not available, and
  claiming it would be dishonest — LMETRIC-alone will always win on pure TTFT/TBT since it
  spends 100% of its attention there).
- Will claim: *at a quantified, honestly-reported latency cost, this achieves a quantified
  ramp-rate/power-correlation reduction* — same tradeoff-disclosure structure already used
  in the sibling PES-IM paper ("lowers ramp rate ~35% at a modest energy cost").
- **Structural, checkable prediction to validate, not assume:** DRF's max-based rule only
  diverges from the compute/load-optimal (LMETRIC) decision when a candidate replica's
  power share is *already* the dominant one for at least one candidate — i.e., the latency
  cost should be **concentrated in power-pressure windows** (e.g., whale-driven ramps), not
  spread as a blanket tax across all traffic. Report latency metrics conditioned on
  power-pressure windows vs. normal windows to test this directly, not just an
  unconditioned average delta.

### 3.3 Evaluation conditions

Three conditions, run on identical workload/hardware:

1. **vLLM default routing** — round-robin across replicas. (vLLM ships no built-in
   multi-replica router of its own — confirmed by inspecting the installed package; the
   external-load-balancer-with-round-robin pattern is the de facto default when nothing
   smarter is deployed, so this is the honest content-blind/state-blind baseline, not a
   literal vLLM component.)
2. **LMETRIC alone** — `P-token × BS`, fully reimplemented (not approximated) per the
   fidelity decision made during design: real per-replica prefix/radix-tree cache-state
   mirror for `P-token`, real lifecycle-tracked batch size for `BS`. Shares its
   implementation machinery with condition 3.
3. **Ours (DRF-3way)** — as in §3.1, reusing condition 2's `P-token`/`BS` tracking plus the
   new power share.

This 3-condition set (not 4 — a naive power-only baseline was considered and dropped to
keep the comparison tight) directly answers: does adding power cost anything relative to
the state-of-the-art baseline (2 vs 1), and does our fairness-based mechanism actually
change routing decisions in the way §3.2 predicts (3 vs 2, broken out by power-pressure
window).

## 4. Execution steps

1. Confirm the real e-Energy 2027 CFP (deadline/track/topics) — still outstanding from
   earlier in this project's history.
2. Decide the ramp-rate ceiling used to normalize `Share_power,i` — anchor to the PES-IM
   gate experiment's measured figures, or calibrate independently. Open item.
3. Confirm whether the routing experiment's workload has cross-request prefix sharing worth
   preserving (multi-turn/repeated system prompts) — if it's single-turn/isolated like the
   longprompt-TBT workload, `Share_compute,i`'s cache-hit computation still applies but its
   practical effect on routing decisions may be small; note this rather than assume either
   way.
4. Launch N vLLM replicas on the 8×4090 fleet (N per the TP/model-size tradeoff already
   scoped: up to 8 for 7B/TP=1, up to 4 for 14B/TP=2).
5. Build the shared machinery first: prefix/radix-tree cache-state mirror, live
   running+queued tracking, NVML power sampling integration (reuse `power_logger.py`).
6. Implement condition 1 (round-robin) and condition 2 (LMETRIC) using that shared
   machinery; sanity-check condition 2 reproduces sensible cache/load-aware behavior before
   adding condition 3.
7. Implement condition 3 (DRF-3way) on top.
8. Run the whale-injection workload through all three conditions.
9. Report: aggregate (summed-across-GPU) power ramp rate/peak per condition (no new
   CF-measurement code — reuse existing analysis conventions); TTFT/TBT deltas for
   condition 3 vs. condition 2, broken out by power-pressure window vs. normal window per
   §3.2's structural prediction.
10. If the power effect and the concentrated-cost prediction both hold, feed the measured
    ramp-rate effect size into the existing (unchanged) Monte Carlo model
    (`scripts/pesim/coincidence_factor_model.py`) for the data-center-scale extrapolation.
11. Replicate (multi-trial) before writing up — small-N noise risk flagged from the start,
    per the sibling PES-IM/MLSys projects' established practice.

## Open items (explicit, not silently assumed)

- Exact e-Energy 2027 CFP deadline/track/topics (step 1).
- Ramp-rate ceiling calibration for `Share_power,i` (step 2).
- Whether cache-affinity materially matters for this specific workload (step 3) — affects
  how much `Share_compute,i` actually influences routing decisions in practice, not whether
  it's implemented.
