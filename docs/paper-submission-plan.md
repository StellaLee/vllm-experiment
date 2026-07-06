# MLSys Paper Submission Plan

**Working title:** Cache-Aware Scheduling for LLM Serving: Eviction, Ordering, and Chunk Control as a Unified Hierarchy  
**Target venue:** MLSys 2027 (main track); MLSys 2026 workshop as interim milestone  
**Repo:** StellaLee/vllm-experiment  

---

## Current Status (as of 2026-07-06)

Three interventions implemented and benchmarked on a single RTX 4090 with Qwen2.5-Coder-7B-Instruct:

| Experiment | Key result | Findings file |
|-----------|-----------|---------------|
| CF eviction policy | CF > LRU on hit rate under pressure | `findings/2026-07-03-full-eviction-comparison.md` |
| Static chunk sweep + dynamic controller | 2048 default is a saddle point; dynamic controller closes gap | `findings/2026-07-06-chunk-sweep.md` (remote) |
| Prefix-aware request reordering | TTFT p50 −26% on ShareGPT; TPOT +35% (expected) | `findings/2026-07-06-reorder-experiment.md` (remote) |

Central hypothesis: the three interventions are complementary and super-additive — CF eviction retains the right blocks, reordering schedules cache-warm requests first, and the dynamic chunk controller prevents decode starvation when prefill surges.

---

## Phase 1 — Complete Single-GPU Story (2–3 weeks)

**Goal:** workshop-ready paper; self-contained on one GPU.

### 1.1 Combined condition experiment
- Run `PREFIX_REORDER=1 DYNAMIC_CHUNK=1` on BurstGPT and ShareGPT
- Expected: TTFT gains from reordering preserved; TPOT degradation absorbed by chunk controller
- This is the paper's headline result

### 1.2 Full ablation matrix
- 2×2: eviction (LRU / CF) × reordering (off / on), static chunk 2048, both workloads
- 2×3: chunk mode (static-256 / static-4096 / dynamic) × reordering (off / on), CF eviction
- Verifies super-additivity; rules out confounds

### 1.3 Prefix caching control
- Run with `--no-enable-prefix-caching`; shows how much reordering benefit depends on prefix cache being active
- Zero new code, one flag

### 1.4 Overhead measurement
- Profile per-step cost of: prefix hash tree lookup (reordering), decode queue check (chunk controller)
- Must be negligible (<1% scheduling overhead) to survive reviewer scrutiny

### 1.5 Clean implementation
- Replace patch scripts with a proper fork/branch of vLLM
- Write unit tests for the scheduler modifications
- Paper cannot describe "monkey-patching"; needs to present as a clean design

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

1. 2048 is a workload-dependent saddle point → static sweep (done)
2. CF eviction improves hit rate under memory pressure → eviction comparison (done)
3. Reordering reduces TTFT by up to 26% on prefix-heavy workloads → reorder experiment (done)
4. Dynamic chunk controller absorbs TPOT penalty from reordering → combined experiment (Phase 1.1)
5. The three are super-additive → ablation matrix (Phase 1.2)
6. Results hold at 32B/70B scale → Phase 2.1
7. Gains are over Sarathi-Serve, not just vLLM default → Phase 3.1

---

## Timeline

| Phase | Target completion | Milestone |
|-------|-------------------|-----------|
| Phase 1 (single-GPU story) | 2026-07-27 | Workshop submission draft |
| MLSys 2026 workshop deadline | ~2026-08 | Submit workshop paper |
| Phase 2 (scale + diversity) | 2026-09-30 | Multi-GPU results in hand |
| Phase 3 (baselines) | 2026-10-15 | Sarathi-Serve comparison done |
| Phase 4 (theory) | 2026-10-31 | Theory section drafted |
| Phase 5 (writing) | 2026-11-30 | Full draft circulated |
| MLSys 2027 submission | ~2026-12 | Submit main track paper |

---

## Open Questions

- **Disaggregation interaction:** does prefix reordering still help when prefill and decode are on separate nodes? If yes, it is strictly complementary; if no, the scope of the claim needs to be bounded.
- **Controller step size:** bang-bang (×2/÷2) is too coarse for ShareGPT; a ±25% additive step or moving-average filter may let the controller reach the static optimum. Worth a short ablation.
- **Eviction + reordering coupling:** CF eviction retains blocks that reordering will prioritize; does the combination achieve higher effective cache hit rate than either alone? Quantify this in the ablation matrix.
- **Prefix hash tree overhead at scale:** lookup is O(prefix_depth); at 32K-token prompts under TP=4, is it still negligible?
