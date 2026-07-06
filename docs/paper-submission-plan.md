# MLSys Paper Submission Plan

**Working title:** Cache-Aware Scheduling for LLM Serving: Eviction, Ordering, and Chunk Control as a Unified Hierarchy  
**Target venue:** MLSys 2027 (main track); MLSys 2026 workshop as interim milestone  
**Repo:** StellaLee/vllm-experiment  

---

## Current Status (as of 2026-07-06)

Four experiments completed on a single RTX 4090 with Qwen2.5-Coder-7B-Instruct:

| Experiment | Key result | Findings file |
|-----------|-----------|---------------|
| CF eviction policy | CF > LRU on hit rate under pressure | `findings/2026-07-03-full-eviction-comparison.md` |
| Static chunk sweep + dynamic controller | 2048 default is a saddle point; dynamic controller closes gap | `findings/2026-07-06-chunk-sweep.md` |
| Prefix-aware request reordering | TTFT p50 −26% on ShareGPT; TPOT +35% (side effect) | `findings/2026-07-06-reorder-experiment.md` |
| **Combined condition** | TTFT p50 −30.2%, TPOT p95 −11.6%, throughput +19.5% on ShareGPT | `findings/2026-07-06-combined-experiment.md` |

Central hypothesis: the three interventions are complementary and super-additive — CF eviction retains the right blocks, reordering schedules cache-warm requests first, and the dynamic chunk controller prevents decode starvation when prefill surges.

**Important caveat:** all 0706 results use `rate=inf` (thundering-herd, 150 requests simultaneous). Gains are amplified by queuing pressure and must be validated under realistic Poisson arrival rates before submission.

---

## Phase 1 — Complete Single-GPU Story (2–3 weeks)

**Goal:** workshop-ready paper; self-contained on one GPU.

### 1.1 Combined condition experiment ✓ DONE
- Ran `PREFIX_REORDER=1 DYNAMIC_CHUNK=1` on BurstGPT and ShareGPT
- Result: TTFT p50 −30.2%, TPOT p95 −11.6%, throughput +19.5% on ShareGPT; every metric improves
- See `findings/2026-07-06-combined-experiment.md`

### 1.2 Ablation studies (see §Ablation Plan below for full detail)
- Full 2×2×2 factorial: eviction × reordering × chunk (4 CF conditions missing)
- Arrival rate sweep: Poisson 1/2/4/8 req/s — validates gains outside `rate=inf`
- Reordering mechanism: per-step re-sort vs. admission-time sort vs. none
- KV cache hit rate logging per condition — validates the causal mechanism

### 1.3 Prefix caching control
- Run with `--no-enable-prefix-caching`; shows reordering gives zero benefit without prefix cache
- Zero new code, one flag

### 1.4 Overhead measurement
- Profile per-step cost of: prefix hash tree lookup (reordering), decode queue check (chunk controller)
- Must be negligible (<1% scheduling overhead) to survive reviewer scrutiny

### 1.5 Clean implementation
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
| 3 | Reordering reduces TTFT on prefix-heavy workloads | Reorder experiment | ✓ done |
| 4 | Chunk controller absorbs TPOT penalty from reordering | Combined experiment | ✓ done |
| 5 | Three interventions are super-additive | Full 2×2×2 factorial (ablation B) | ✗ partial |
| 6 | Gains persist under realistic arrival rates | Poisson rate sweep (ablation A) | ✗ needed |
| 7 | Reordering benefit requires prefix caching | `--no-enable-prefix-caching` run | ✗ needed |
| 8 | Per-step re-sort outperforms admission-time sort | Mechanism ablation (ablation C) | ✗ needed |
| 9 | Coupling is mechanistic (via hit rate) | Hit rate logging per condition (ablation D) | ✗ needed |
| 10 | Gains exceed Sarathi-Serve tuned static chunk | Sarathi baseline (ablation E) | ✗ needed |
| 11 | Results hold at 32B/70B scale | Multi-GPU experiments (Phase 2.1) | ✗ needed |
| 12 | System outperforms or matches PRISM | PRISM baseline (ablation F) | ✗ needed |

---

## Timeline

| Phase | Target completion | Milestone |
|-------|-------------------|-----------|
| Tier 1 ablations (A–D) | 2026-07-13 | ~3 hrs compute; closes workshop argument |
| Phase 1 complete | 2026-07-20 | Clean implementation + overhead measurement |
| Workshop draft | 2026-07-27 | Claims 1–9 substantiated |
| MLSys 2026 workshop deadline | ~2026-08 | Submit workshop paper |
| Tier 2 ablations (E–G) | 2026-09-15 | Sarathi + PRISM baselines in hand |
| Phase 2 (scale + diversity) | 2026-09-30 | Multi-GPU results |
| Phase 3 (remaining baselines) | 2026-10-15 | Disaggregation framing done |
| Phase 4 (theory) | 2026-10-31 | Theory section drafted |
| Phase 5 (writing) | 2026-11-30 | Full draft circulated |
| MLSys 2027 submission | ~2026-12 | Submit main track paper |

---

## Open Questions

- **Gains at realistic load:** `rate=inf` amplifies queuing effects; gains may shrink substantially under Poisson arrivals. This is the single biggest risk to the paper's practical claim. Ablation A answers it.
- **Disaggregation interaction:** does prefix reordering still help when prefill and decode are on separate nodes? If yes, strictly complementary; if no, scope of claim must be bounded.
- **Controller step size:** bang-bang (×2/÷2) is too coarse for ShareGPT; a ±25% additive step or moving-average filter may reach the static optimum more consistently. Ablation G answers it.
- **Eviction + reordering coupling mechanism:** CF retains blocks that reordering will prioritise — does the combination achieve higher effective hit rate than either alone? Ablation D (hit rate logging) quantifies this.
- **PRISM overlap risk:** PRISM (arXiv:2605.08581) co-designs scheduling + eviction and reports similar TTFT gains. If their system also naturally suppresses TPOT degradation, our third layer (chunk control) loses novelty. Must run PRISM as a direct baseline.
- **Prefix hash tree overhead at scale:** lookup is O(prefix_depth); at 32K-token prompts under TP=4, overhead needs measurement.
- **Statistical validity:** all experiments use 150 prompts at `rate=inf`. Small sample size inflates variance. The +19.5% throughput on ShareGPT in particular warrants a repeat run to confirm.
