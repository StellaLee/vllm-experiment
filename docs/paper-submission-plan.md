# MLSys Paper Submission Plan

**Working title:** Decode-Starvation-Aware Chunk Control for Prefix-Cached LLM Serving  
**Target venue:** MLSys 2027 (main track); MLSys 2026 workshop as interim milestone  
**Repo:** StellaLee/vllm-experiment  

---

## Current Status (as of 2026-07-07)

Five experiments completed on a single RTX 4090 with Qwen2.5-Coder-7B-Instruct:

| Experiment | Key result | Findings file |
|-----------|-----------|---------------|
| CF eviction policy | CF > LRU on hit rate under pressure | `findings/2026-07-03-full-eviction-comparison.md` |
| Static chunk sweep + dynamic controller | 2048 default is a saddle point; dynamic controller closes gap | `findings/2026-07-06-chunk-sweep.md` |
| Prefix-aware request reordering | TTFT p50 −26% on ShareGPT at rate=inf; TPOT +35% (side effect) | `findings/2026-07-06-reorder-experiment.md` |
| **Combined condition** | TTFT p50 −30.2%, TPOT p95 −11.6%, throughput +19.5% on ShareGPT at rate=inf | `findings/2026-07-06-combined-experiment.md` |
| **Arrival rate sweep (Ablation A)** | TTFT gains do NOT hold at realistic rates; TPOT/E2EL tail robust across all rates | `findings/2026-07-07-rate-sweep.md` |

**Revised central hypothesis:** Prefix-aware request reordering induces decode starvation under high utilization, measurably degrading TPOT. A decode-queue-depth-driven chunk controller absorbs this penalty and provides robust TPOT and tail latency improvements across all arrival rates. Combined with prefix-aware scheduling, the system achieves super-additive TTFT gains specifically near saturation.

**Key scope revision from rate sweep:** TTFT gains (−30.2%) were a `rate=inf` queuing artifact. At sub-saturation Poisson rates (1–4 req/s), TTFT delta is near zero. TPOT and E2EL tail improvements are genuine and rate-robust. The paper's primary claim must shift to **TPOT/tail latency protection**, with TTFT improvement framed as a saturation-regime bonus. Starvation at 8 req/s BurstGPT (+24% TTFT) confirms an aging mechanism is required before submission.

---

## Phase 1 — Complete Single-GPU Story (2–3 weeks)

**Goal:** workshop-ready paper; self-contained on one GPU.

### 1.1 Combined condition experiment ✓ DONE
- Ran `PREFIX_REORDER=1 DYNAMIC_CHUNK=1` on BurstGPT and ShareGPT
- Result: TTFT p50 −30.2%, TPOT p95 −11.6%, throughput +19.5% on ShareGPT; every metric improves
- See `findings/2026-07-06-combined-experiment.md`

### 1.2 Aging mechanism for reordering *(now mandatory)*
- Add max-wait threshold: promote any request waiting >T ms regardless of prefix warmth
- Without this, reordering degrades TTFT at high load (BurstGPT 8 req/s: +24%)
- Measure: TTFT distribution for warm vs. cold requests; show starvation eliminated
- Effort: ~2 hours of code + 1 hour benchmarking

### 1.3 Near-saturation experiment *(new, replaces rate=inf as headline)*
- Run combined condition at arrival rates that bring system to ~80–95% utilization
- ShareGPT saturation ≈ 5 req/s → test at 4–5 req/s; BurstGPT ≈ 4 req/s → test at 3–4 req/s
- This is the regime where gains are real and meaningful; should be the paper's primary figure
- Effort: ~1 hour, zero new code

### 1.4 Ablation studies (see §Ablation Plan below for full detail)
- Full 2×2×2 factorial: eviction × reordering × chunk (4 CF conditions missing)
- Arrival rate sweep: ✓ DONE — see `findings/2026-07-07-rate-sweep.md`
- Reordering mechanism: per-step re-sort vs. admission-time sort vs. none
- KV cache hit rate logging per condition — validates the causal mechanism

### 1.5 Prefix caching control
- Run with `--no-enable-prefix-caching`; shows reordering gives zero benefit without prefix cache
- Zero new code, one flag

### 1.6 Overhead measurement
- Profile per-step cost of: prefix hash tree lookup (reordering), decode queue check (chunk controller)
- Must be negligible (<1% scheduling overhead) to survive reviewer scrutiny

### 1.7 Clean implementation
- Replace patch scripts with a proper fork/branch of vLLM
- Write unit tests for the scheduler modifications
- Paper cannot describe "monkey-patching"; needs to present as a clean design

---

## Ablation Plan

Ablations are structured to address three reviewer attack vectors: (1) "PRISM/CacheWise already does this," (2) "your gains disappear under realistic load," and (3) "your design choices are arbitrary."

### Tier 1 — Required for workshop submission (~3 hours total compute)

**A. Arrival rate sweep** *(most critical)*
- Run combined condition at Poisson rates 1, 2, 4, 8 req/s in addition to existing `rate=inf`
- Plot TTFT and TPOT deltas vs. baseline as a function of arrival rate
- Expected: gains shrink at low load (less queuing), persist at moderate load
- If gains disappear at ≤2 req/s the contribution is narrowly scoped to burst scenarios
- Effort: ~1 hour, zero new code

**B. Full 2×2×2 factorial** *(closes the super-additivity argument)*

All 8 cells of {eviction: LRU/CF} × {reordering: off/on} × {chunk: static/dynamic}:

| Eviction | Reorder | Chunk | Tag | Status |
|----------|---------|-------|-----|--------|
| LRU | off | static 2048 | baseline | ✓ done |
| LRU | off | dynamic | dynamic_only | ✓ done |
| LRU | on | static 2048 | reorder_only | ✓ done |
| LRU | on | dynamic | combined | ✓ done |
| CF | off | static 2048 | cf_baseline | ✗ needed |
| CF | off | dynamic | cf_dynamic | ✗ needed |
| CF | on | static 2048 | cf_reorder | ✗ needed |
| CF | on | dynamic | cf_combined | ✗ needed |

Four CF conditions are missing; adding them closes the factorial and measures whether CF eviction is itself super-additive with the scheduling interventions. Effort: ~40 min, one new script.

**C. Reordering mechanism comparison** *(addresses FEATHER "why not RL?" objection)*
- Three variants: none / admission-time sort (once at arrival) / per-step re-sort (current)
- If per-step beats admission-time: proves dynamic reordering matters as cache state evolves
- If equal: simpler approach suffices, weakens the RL argument further either way
- Effort: ~30 min, one flag in the scheduler patch

**D. KV cache hit rate per condition** *(validates causal mechanism)*
- Log KV hit rate for all 8 factorial conditions (vLLM already exposes this metric)
- Show: hit rate increases reorder-only → CF+reorder; TTFT improvement correlates with hit rate
- Turns super-additivity from a correlation into a mechanistic claim
- Effort: ~1 hour, add hit rate column to analyzer scripts

### Tier 2 — Required for main track

**E. Sarathi-Serve baseline**
- Run Sarathi's recommended static chunk size on the same workloads
- Show dynamic controller matches or beats offline-tuned value without workload foreknowledge
- Effort: ~2 hours (config change + benchmark)

**F. PRISM as a direct baseline**
- PRISM co-designs scheduling + eviction (arXiv:2605.08581); closest prior work
- Must show combined system either outperforms PRISM or matches it with lower complexity
- Effort: 1–2 weeks to implement faithfully; or use reported numbers if code is public

**G. Chunk controller design ablation**
- Compare bang-bang ×2/÷2 (current) vs. additive ±25% vs. exponential moving average
- Justifies design choice; shows sensitivity to step size
- Effort: ~3 hours, small code change

### Tier 3 — Strengthening (main track polish)

**H. Sensitivity to max-num-seqs** (16/32/64) — prevents "specific to max-seqs=32" objection  
**I. GPU memory utilization sweep** (0.7/0.8/0.9) — more pressure → larger CF eviction advantage  
**J. Prefix caching on/off control** — `--no-enable-prefix-caching` shows reordering needs prefix cache to work

### Ablation priority for execution

| Priority | Ablation | Effort | Blocks |
|----------|----------|--------|--------|
| 1 | Arrival rate sweep (1/2/4/8 req/s) | 1 hr | Workshop |
| 2 | Full 2×2×2 factorial (4 CF conditions) | 40 min | Workshop |
| 3 | Admission-time vs. per-step reorder | 30 min | FEATHER rebuttal |
| 4 | KV hit rate logging per condition | 1 hr | Mechanism claim |
| 5 | Sarathi-Serve baseline | 2 hr | Main track |
| 6 | Chunk controller step size | 3 hr | Main track |
| 7 | PRISM baseline | 1–2 wk | Main track |
| 8 | Sensitivity sweeps (H/I/J) | 2 hr | Main track polish |

---

## Phase 2 — Scale and Model Diversity (4–6 weeks)

**Goal:** main track credibility; eliminates single-model / single-GPU objections.

### 2.1 Multi-GPU experiments
- TP=2 and TP=4 on Qwen2.5-32B or Llama-3-70B (rent A100 node)
- Key question: does chunk sizing interact with inter-GPU communication overhead?
- Reordering under TP: prefix hash lookup is local to rank 0 scheduler — verify no distributed side effects

### 2.2 Model diversity
- Minimum: 7B + 32B + 70B, same family
- Stretch: second family (Llama vs. Qwen) to rule out model-specific artifacts
- Show that chunk size optimum shifts with model size (larger KV footprint per block)

### 2.3 Long-context workload
- RAG or document QA with 8K–32K prompts
- Stress-tests eviction (large KV footprint) and prefix reuse (repeated document prefixes)
- BurstGPT + ShareGPT alone are too short-context for a 2027 paper

### 2.4 Production traces
- Azure LLM public trace dataset
- Realistic arrival patterns (Poisson, bursty) rather than `rate=inf` thundering herd
- Poisson arrival rate sweep: 1, 2, 4, 8 req/s

---

## Phase 3 — Baselines (3–4 weeks)

**Goal:** situate the work relative to published systems; address "why not just use X?" reviewers.

### 3.1 Sarathi-Serve
- Closest prior work on chunked prefill
- Reproduce their chunk size sweep; show dynamic controller matches or beats their tuned static value across workloads
- Key differentiator: Sarathi fixes chunk size offline; we adapt online

### 3.2 Orca
- Iteration-level scheduling baseline
- Comparison establishes that our gains are over a strong scheduling baseline, not just over naive batching

### 3.3 Disaggregation baseline (Splitwise / Mooncake framing)
- The current industry direction; reviewers will ask "why not disaggregate prefill and decode?"
- Argument: disaggregation requires dedicated prefill hardware; our approach gives most of the benefit within a single serving instance at zero hardware cost
- Show conditions where scheduling-side wins (low-to-moderate load) vs. where disaggregation wins (saturated load)

---

## Phase 4 — Theory (3–4 weeks, parallel with Phase 2)

**Goal:** meet MLSys bar for analytical contribution; gives reviewers something to cite.

### 4.1 Queue-theoretic model
- Model waiting queue as M/G/1 with prefix-cache-dependent service time
- Show analytically that reordering reduces mean TTFT by a factor proportional to cache hit rate
- Validate empirically: hit rate × predicted reduction should match observed −26% on ShareGPT

### 4.2 Convergence analysis for bang-bang controller
- Under what arrival distribution does the controller converge to the static optimum?
- Key reviewer question: "why not just tune the chunk size offline?"
- Answer: the optimal static value is workload-dependent (BurstGPT optimum is 4096, ShareGPT optimum is 256); convergence proof shows the controller reaches near-optimal without workload foreknowledge

### 4.3 Super-additivity bound
- Formal condition under which combining eviction + reordering + chunk control is strictly better than any single intervention
- Even a partial result (e.g., under independence assumptions) strengthens the coupling hypothesis

---

## Phase 5 — Writing (3–4 weeks)

### Paper structure

| Section | Content | Owner |
|---------|---------|-------|
| Abstract | Three interventions, headline numbers, one-sentence coupling claim | — |
| Introduction | Motivation: KV cache hit rate × scheduling order interaction; 2048 saddle point as empirical hook | — |
| Background | vLLM scheduler, prefix caching, chunked prefill, eviction policies | — |
| Problem formulation | Online scheduling with prefix cache state; TTFT/TPOT Pareto frontier as objective | — |
| System design | Three-layer hierarchy: eviction → reordering → chunk control; each layer's mechanism | — |
| Theory | Queue model; convergence bound; super-additivity condition | — |
| Evaluation | §6.1 single-GPU ablations; §6.2 multi-GPU scale; §6.3 production traces; §6.4 overhead | — |
| Related work | Sarathi-Serve, Orca, Splitwise, TetriInfer, FlexGen, vLLM | — |
| Conclusion | Coupling hypothesis confirmed; open problems (optimal policy, disaggregation interaction) | — |

### Key claims to substantiate

| # | Claim | Evidence needed | Status |
|---|-------|----------------|--------|
| 1 | 2048 is a workload-dependent saddle point | Static chunk sweep | ✓ done |
| 2 | CF eviction improves hit rate under pressure | Eviction comparison | ✓ done |
| 3 | Reordering reduces TTFT **at high utilization** | Reorder + rate sweep | ✓ done (scoped) |
| 4 | Chunk controller absorbs TPOT penalty from reordering | Combined experiment | ✓ done |
| 5 | TPOT/tail improvements are robust across arrival rates | Rate sweep (ablation A) | ✓ done |
| 6 | Reordering degrades TTFT at high load without aging | Rate sweep 8 req/s BurstGPT | ✓ done |
| 7 | Aging mechanism eliminates starvation safely | Aging experiment (Phase 1.2) | ✗ needed |
| 8 | Near-saturation is the right operating regime for gains | Near-saturation experiment (Phase 1.3) | ✗ needed |
| 9 | Three interventions are super-additive | Full 2×2×2 factorial (ablation B) | ✗ partial |
| 10 | Reordering benefit requires prefix caching | `--no-enable-prefix-caching` run | ✗ needed |
| 11 | Per-step re-sort outperforms admission-time sort | Mechanism ablation (ablation C) | ✗ needed |
| 12 | Coupling is mechanistic (via hit rate) | Hit rate logging per condition (ablation D) | ✗ needed |
| 13 | Gains exceed Sarathi-Serve tuned static chunk | Sarathi baseline (ablation E) | ✗ needed |
| 14 | Results hold at 32B/70B scale | Multi-GPU experiments (Phase 2.1) | ✗ needed |
| 15 | System outperforms or matches PRISM | PRISM baseline (ablation F) | ✗ needed |

---

## Timeline

**MLSys 2027 submission deadline: October 2026. All phases must complete by end of September.**

| Phase | Target completion | Milestone |
|-------|-------------------|-----------|
| Aging mechanism + near-saturation experiment | 2026-07-10 | Starvation fixed; new headline figure |
| Tier 1 ablations (B–D) | 2026-07-14 | 2×2×2 factorial + hit rate logging |
| Phase 1 complete | 2026-07-21 | Clean implementation + overhead |
| Workshop draft | 2026-07-28 | Claims 1–12 substantiated; reframed thesis |
| MLSys 2026 workshop deadline | ~2026-08 | Submit workshop paper |
| Tier 2 ablations (E–G) | 2026-08-31 | Sarathi + PRISM baselines in hand |
| Phase 2 (scale + diversity) | 2026-09-10 | Multi-GPU results — **must overlap with writing** |
| Phase 3 (remaining baselines) | 2026-09-20 | Disaggregation framing done |
| Phase 4 (theory) | 2026-09-25 | Theory section drafted — can be lightweight for main track |
| Phase 5 (writing) | 2026-10-01 | Full draft circulated for review |
| **MLSys 2027 submission** | **2026-10** | **Submit main track paper** |

**Timeline risk:** Phases 2–5 compress into 6 weeks (mid-August to October). Multi-GPU experiments (Phase 2) require renting A100 nodes and take calendar time regardless of effort. If PRISM baseline (ablation F) is not public by end of August, it falls off the main track submission. Theory (Phase 4) should be scoped to a short analytical section rather than a full formal treatment to fit the timeline.

---

## Open Questions

- **Gains at realistic load:** ✓ ANSWERED by rate sweep. TTFT gains don't persist below saturation; TPOT/tail gains do. Paper scope is now explicitly bounded to high-utilization regime.
- **Starvation at high load:** ✓ CONFIRMED at 8 req/s BurstGPT (+24% TTFT). Aging mechanism required before submission.
- **Near-saturation operating point:** What arrival rate brings each workload to 80–90% utilization? That is the regime where all three interventions contribute; it should be the paper's primary evaluation point.
- **Aging threshold tuning:** what wait threshold T balances starvation prevention vs. TTFT gain? Too low T → no reordering benefit; too high T → starvation persists. Needs empirical sweep.
- **Disaggregation interaction:** does prefix reordering still help when prefill and decode are on separate nodes? If yes, strictly complementary; if no, scope of claim must be bounded.
- **Controller step size:** bang-bang (×2/÷2) is too coarse; a ±25% additive step may be more stable. Ablation G answers it.
- **PRISM overlap risk:** PRISM uses static chunk size and doesn't measure TPOT — our chunk controller is the differentiator. Confirmed by literature review. Must still run PRISM as a direct baseline for main track.
- **Prefix hash tree overhead at scale:** lookup is O(prefix_depth); at 32K-token prompts under TP=4, needs measurement.
- **Statistical validity:** 150 prompts per run; near-saturation results may have higher variance. Repeat runs needed for the primary near-saturation figure.
