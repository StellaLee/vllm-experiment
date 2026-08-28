# Related Work & Novelty Assessment

**Last updated:** 2026-08-26
**Supersedes:** the 2026-07-06 version of this file (and `literature-review-draft.md`,
now marked superseded), which mapped positioning for the abandoned three-layer
eviction+reordering+chunk-controller paper. This version maps positioning for the
current paper (`paper-mlsys/paper.md`, "When Prefix-Aware Serving Wins Are Real—and
When They're Benchmarking Artifacts"): the genuine-vs-artifact methodology critique,
the whale-scale boundary condition for chunking's decode-protection win, the
per-request adaptive-cap controller, and the decode-protection/admission-sharing
tradeoff (§5.8).

**Purpose:** track what's already cited in the live tex vs. what a 2026-08-26 literature
pass surfaced that isn't cited yet, so nothing a reviewer would raise gets missed.

---

## Already cited in the live paper

These are correctly positioned in `paper-mlsys/tex/sections/02-background.tex` and
`07-protocol.tex` — listed here only so this doc is a complete map, not because they
need rework.

| Paper | Cite key | Role in our positioning |
|---|---|---|
| Sarathi-Serve (arXiv:2403.02310) | `agrawal2024sarathi` | Our chunking mechanism is a dynamic generalization of its fixed budget; our `static-512` baseline *is* Sarathi's recommended setting. §7.1 shows the dynamic controller doesn't beat this baseline in the decode-bound open-loop regime. |
| PRISM (arXiv:2605.08581) | `prism` | Closest prior prefix-aware scheduler (QAS). §7.1: PRISM's own open-loop TTFT gain is a *caching* effect (orthogonal to the artifact we study), and its cold-lane reservation independently corroborates our conservation-law account. |
| Preble (ICLR 2025) | `preble` | Cited as evidence the research frontier already evaluates correctly (open-loop / trace replay), unlike common benchmark-harness defaults. |
| vLLM `benchmark_serving` | `vllmbench` | The concrete tool whose `--request-rate inf` default *is* the synchronized-start herd we show produces an artifact win. |
| "Beyond Prediction: Tail-Aware Scheduling for LLM Inference" (arXiv:2606.18431) | `tailaware` | **Bib entry was a placeholder (`{TODO}` author, unverified title) — fixed as part of this pass; real title/author confirmed.** Cited as the broader tail-aware-scheduling family sharing the same arrival-model exposure our protocol addresses. |

---

## Found in the 2026-08-26 literature pass, not yet cited — action needed

### 1. Fairness-Aware and Latency-Controllable Scheduling for Chunked-Prefill LLM Serving (arXiv:2606.09061)

**The tightest overlap by title and mechanism.** Replaces static chunk budgets with
Latency-Prediction-Based Request Scheduling (LPRS) and Active Prefill Control (APC) —
a target-time-driven controller that actively regulates prefill concurrency, plus an
aging-based fairness policy. Reports >10% mean E2E latency reduction over FCFS and
P99 tail reduction over static budgets.

**Distinction from our work:**
- Their controller is *latency-prediction-based* (needs a target-time estimate per
  request); ours is a *per-request length cap* (§5.7) requiring no prediction, and a
  separate step-budget controller (§5)  — neither needs a latency target.
- They report no benchmarking-methodology critique — no closed-loop-vs-open-loop
  artifact analysis. Our core claim (the herd win is a synchronized-start artifact)
  is untouched by this paper.
- They report no whale-scale/rarity boundary condition (§ nowhale finding) and no
  decode-protection-vs-admission-sharing tradeoff (§5.8). Their evaluation doesn't
  disaggregate a bursty long-tail population from a short-request majority the way
  ours does.
- **Risk: high — if a reviewer finds one paper to compare us against, it's this one.**
  Must cite and explicitly distinguish in §7.1.

### 2. From Tokens to Layers: Redefining Stall-Free Scheduling for MoE Serving with Layered Prefill (arXiv:2510.08055, accepted MLSys 2026)

States "chunked prefill... effective at stabilizing TBT" as an *established baseline
fact* in its own introduction, then targets a different problem (MoE expert-weight
reload overhead from chunking, via layer-granularity scheduling instead of
token-granularity). Not a competitor on our axis (single-dense-model, prefill/decode
interleaving), but important because:
- It confirms the *base* claim ("chunking stabilizes TBT") is now conventional
  wisdom at MLSys, not a claim we can present as novel on its own — we already do
  not lead with this (our contribution is the artifact/boundary/tradeoff triad), but
  this paper is good evidence for that framing choice and worth citing as such.
- **Action: cite as evidence the field treats TBT-stabilization as established,
  motivating why our contribution must be the methodology/boundary/tradeoff findings
  and not the base mechanism.**

### 3. CascadeInfer: Length-Aware Scheduling of LLM Serving with Low Latency and Load Balancing (arXiv:2512.19179)

Already uses "whale"/"minnow" terminology for long/short requests — so that
vocabulary is not ours to claim as novel. Partitions serving *instances* into
length-specialized groups and migrates KV cache across them at the fleet/routing
layer.

**Distinction from our work:** mechanically orthogonal — instance-level physical
separation of whale/short traffic vs. our single-instance temporal interleaving via
chunk budget. A deployment could plausibly use both together (route whales to
dedicated instances *and* chunk within an instance for the whale/short mix that still
lands together). Worth one sentence in §7.1 to preempt "why not just route whales
elsewhere" as a reviewer question — the answer is CascadeInfer requires provisioning
dedicated whale capacity; our mechanism works within a single shared instance at zero
extra hardware, the same argument already made for chunking vs. disaggregation.

### 4. Beyond Greedy Chunking / SlidingServe (arXiv:2606.05933)

Already discussed in the (now-superseded) `literature-review-draft.md` from the old
paper direction, but **never actually cited in the live tex** — a real gap since it's
adaptive chunk sizing driven by TBT-slack-to-deadline, the most directly comparable
prior adaptive-chunking mechanism to our dynamic controller (§5) and adaptive cap
(§5.7).

**Distinction from our work:** SlidingServe requires per-request SLO deadlines as
input (ternary search over TBT slack to a target); our dynamic controller (§5) uses
decode-queue/iteration-latency feedback with no deadline input, and our adaptive cap
(§5.7) uses only the request's own prefill length — neither needs a workload- or
request-level SLO to be specified in advance. SlidingServe is not evaluated for a
closed-loop-artifact effect or a whale-scale boundary condition.

---

## Novelty framing (current paper, current competitor set)

| Dimension | Us | Sarathi-Serve | PRISM | Fairness-Aware (2606.09061) | SlidingServe | CascadeInfer | Layered Prefill |
|---|---|---|---|---|---|---|---|
| Chunked prefill / TBT stabilization | ✓ | ✓ (static) | ✗ | ✓ (predicted) | ✓ (SLO-slack) | ✗ | ✓ (layer-granular) |
| Closed-loop-artifact critique | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Whale-scale/rarity boundary condition characterized | ✓ | ✗ | ✗ | ✗ | ✗ | ✓ (assumes, doesn't characterize the boundary) | ✗ |
| Per-request adaptive cap, no SLO/prediction input | ✓ | ✗ | ✗ | ✗ (needs prediction) | ✗ (needs deadline) | ✗ | ✗ |
| Decode-protection vs. admission-sharing tradeoff reported | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |

**No prior paper combines the artifact critique, the boundary-condition
characterization, and the tradeoff finding.** That three-part combination — not
"chunking protects TBT" alone, which is now conventional wisdom (Layered Prefill) —
is the paper's actual novelty claim.

---

## Action items for `paper-mlsys/`

1. `tex/refs.bib`: fix `tailaware` entry (real author list, currently `{TODO}`); add
   `fairnesslatency` (2606.09061), `layeredprefill` (2510.08055), `cascadeinfer`
   (2512.19179), `slidingserve` (2606.05933).
2. `tex/sections/07-protocol.tex` §7.1: add a paragraph distinguishing from
   Fairness-Aware/SlidingServe (adaptive-chunking family) and one sentence on
   CascadeInfer (why not just route whales elsewhere).
3. Re-run the "papers to read in full" pass before submission — Fairness-Aware
   (2606.09061) specifically, since it's the highest-risk overlap.
