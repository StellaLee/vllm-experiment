# Related Work & Novelty Assessment

**Last updated:** 2026-07-06  
**Purpose:** Map existing literature against our contributions; identify what's novel and what needs repositioning.

---

## Papers That Overlap Our Work

### Eviction Policy (CF contribution)

**[Recency/Frequency Adaptive KV Caching](https://arxiv.org/pdf/2606.21238)**  
arXiv:2606.21238 · June 2026  
Proposes a hybrid eviction policy that dynamically balances recency and frequency rather than pure LRU — exactly the same idea as our CF policy. Reports +10.8% KV cache hit rate and −12.6% TTFT over vanilla vLLM.  
**Risk: direct overlap. CF eviction alone is not publishable novelty.**

---

### Prefix-Aware Scheduling (reordering contribution)

**[PRISM: Fast Online LLM Serving via Scheduling-Memory Co-design](https://arxiv.org/pdf/2605.08581)**  
arXiv:2605.08581 · May 2026  
Co-designs a query-aware scheduler (QAS) with a demand-aware radix tree (DART), explicitly aligning request admission with prefix KV retention. Reports −23–37% P99 TTFT. The closest single paper to our overall work.  
**Risk: high. Covers scheduling + eviction co-design; needed as a strong baseline.**

**[Requests of a Feather Must Flock Together](https://arxiv.org/pdf/2605.06046)**  
arXiv:2605.06046 · May 2026  
FEATHER: RL-based scheduler that groups prefix-homogeneous requests; reports 2–10× throughput. More sophisticated than our greedy per-step sort, but the core insight (group requests by prefix similarity) is the same.  
**Risk: medium. Our reordering is simpler; frame as "RL is unnecessary — greedy suffices at single-node scale."**

**[AlignedServe: Prefix-aware Batching](https://arxiv.org/pdf/2605.23389)**  
arXiv:2605.23389 · May 2026  
Groups requests by similar prefix length to eliminate iteration-level bubbles in a disaggregated architecture.  
**Risk: low-medium. Focuses on batching structure, not per-step queue resorting.**

**[CacheWise: KVCache Management for LLM Coding Agents](https://arxiv.org/pdf/2606.16824)**  
arXiv:2606.16824 · June 2026  
Combines prefix-aware scheduling with reuse-aware eviction for coding agent workloads. Two-layer co-design closest to our eviction + reordering combination, but domain-specific (coding agents).  
**Risk: medium. Covers two of our three layers; no chunk controller; no decode-starvation analysis.**

---

### Dynamic Chunk Control

**[Sarathi-Serve: Chunked Prefill](https://arxiv.org/pdf/2403.02310)**  
arXiv:2403.02310 · 2024  
Original chunked prefill paper; static chunk size. The mandatory baseline for our chunk sizing work.

**[Niyama: Breaking the Silos of LLM Inference Serving](https://arxiv.org/pdf/2503.22562)**  
arXiv:2503.22562 · March 2025  
Dynamic chunking to maximise chunk size while meeting request deadlines (SLO-aware).  
**Risk: low. Deadline-driven; our controller responds to decode queue depth, not deadlines.**

**[Beyond Greedy Chunking: SLO-Aware Sliding-Window Scheduling](https://arxiv.org/pdf/2606.05933)**  
arXiv:2606.05933 · June 2026  
SLO-aware dynamic chunk adjustment with a sliding-window scheduler.  
**Risk: low-medium. SLO-driven; our bang-bang controller is load-driven and specifically responds to reordering-induced starvation.**

---

## What Remains Genuinely Novel

### 1. The three-way coupling and TPOT-protection mechanism

No existing paper combines all three layers or identifies the specific failure-mode chain:

> reordering → decode starvation → TPOT degradation  
> chunk controller → detects growing decode queue → shrinks budget → absorbs penalty  
> combined → Pareto-dominant over any single intervention

CacheWise is closest (scheduling + eviction) but has no chunk controller and no decode-starvation analysis. PRISM co-designs scheduling + eviction but has no adaptive chunk budget. Neither reports super-additivity.

### 2. Super-additivity experimental confirmation

Combined condition (PREFIX_REORDER=1 + DYNAMIC_CHUNK=1) outperforms every single-intervention condition on every metric in both workloads:
- ShareGPT TTFT p50: −30.2% (> reorder-only −26.2% > dynamic-only −20.5%)
- ShareGPT TPOT p95: −11.6% (reorder-only was +34.7% — fully reversed)
- ShareGPT throughput: +19.5%
- BurstGPT: every metric improves vs. baseline

### 3. Saddle-point finding on chunk size

The vLLM 2048 default is suboptimal in both directions on ShareGPT — both 256 and 4096 improve TTFT — meaning no static value dominates across workloads. This is a clean empirical motivation for adaptive control that no prior paper has stated directly.

### 4. Decode-starvation as the explicit coupling mechanism

The TPOT degradation caused by reordering is precisely the failure mode the chunk controller is designed to address. This feedback loop (intervention A creates a problem that intervention B solves) is the paper's structural argument. Existing co-design papers (PRISM, CacheWise) optimise the two components jointly without this causal framing.

---

## Revised Paper Positioning

| Original claim | Status | Revised framing |
|----------------|--------|----------------|
| "CF eviction improves KV cache hit rate" | Overlapped by 2606.21238 | Cite as motivation; drop as standalone contribution |
| "Prefix reordering reduces TTFT" | Partially overlapped by PRISM, FEATHER | "Greedy per-step reordering at single-node scale is sufficient — no RL, no disaggregation required" |
| "Dynamic chunk controller adapts to load" | Partially overlapped by Niyama, 2606.05933 | Keep; the decode-starvation response is new |
| "Three interventions are complementary" | **Not covered** | **This is the paper's core claim** |

**Surviving thesis:** *Existing systems optimise scheduling and eviction independently. We show that scheduling-induced TPOT degradation creates a feedback loop that only a decode-aware chunk controller can break, and that the three interventions together are super-additive in a way no single-intervention system has measured.*

---

## Baseline Requirements for Main Track

To beat PRISM and CacheWise as baselines (required for MLSys main track):

1. Reproduce PRISM's QAS + DART on our hardware (or use their reported numbers)
2. Show combined condition outperforms PRISM on TTFT while also improving TPOT
3. Show CacheWise's two-layer approach is strictly dominated by our three-layer system
4. Argue that greedy reordering + bang-bang chunk control is lower-complexity than PRISM's radix-tree co-design

---

## Papers to Read in Full

Priority order:

1. PRISM (arXiv:2605.08581) — must understand their QAS mechanism precisely
2. CacheWise (arXiv:2606.16824) — must understand their scheduling + eviction interaction
3. FEATHER (arXiv:2605.06046) — understand RL formulation; argue why greedy suffices
4. 2606.21238 — understand their ARC-based eviction; map to our CF policy
5. Beyond Greedy Chunking (arXiv:2606.05933) — understand SLO framing vs. our load framing
