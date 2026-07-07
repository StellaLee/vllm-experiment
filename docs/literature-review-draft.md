# Literature Review — Draft for Paper Section

**Last updated:** 2026-07-07  
**Purpose:** Starting point for the Related Work section. Covers all papers read in full or in detail. Includes per-paper mechanism summaries, overlap analysis, and positioning statement.

---

## 1. Chunked Prefill

**Sarathi-Serve (arXiv:2403.02310, 2024)**  
Introduces chunked prefill: instead of processing a full prompt in one step, the prompt is split into fixed-size chunks interleaved with decode iterations. This prevents decode starvation from long prefills and improves TTFT for queued requests. Chunk size is **static** and must be tuned offline per workload. Our dynamic chunk controller is a direct extension of this baseline — we replace the static size with an online controller that adapts to decode queue depth without workload foreknowledge.

**Beyond Greedy Chunking / SlidingServe (arXiv:2606.05933, 2026)**  
Proposes SlidingChunker: an adaptive chunk size policy that uses ternary search over two consecutive scheduling windows to minimize total predicted latency. The chunk budget is driven by **TBT (time-between-tokens) slack** — the minimum remaining time to deadline across all active decode requests. The scheduler (Multi-Level Priority Sorter) orders requests by urgency (remaining work / slack to SLO deadline) and shortest remaining work; it does **not** use prefix cache state as a scheduling signal.

*Overlap with our work:* SlidingChunker demonstrates that adaptive chunk sizing can protect decode throughput. Our dynamic chunk controller serves the same protective function but uses **decode queue depth** as the feedback signal rather than per-request SLO deadlines. This makes our controller self-tuning and deployable without pre-specified latency targets. Critically, SlidingServe's scheduler is deadline-driven and has no connection to prefix caching — the paper does not identify or address the starvation pattern induced by prefix-aware scheduling.

---

## 2. Prefix-Aware Scheduling

**PRISM (arXiv:2605.08581, 2026)**  
Co-designs a Query-Aware Scheduler (QAS) with a Demand-Aware Radix Tree (DART). QAS scores reusable KV segments by demand (weighted sum of queued, active, and next-batch request counts), groups requests sharing top-priority segments into the same scheduling window (hot lane), and maintains a FIFO cold lane to prevent starvation. DART retains KV blocks that QAS will prioritize. Chunk size is **static** (inherited from SGLang defaults). TPOT is not measured; the paper focuses exclusively on TTFT. Decode starvation is mentioned only in a theoretical appendix and not treated as a primary problem. Reports −23% to −37% P99 TTFT on Qwen3-4B and Llama2-13B.

*Overlap with our work:* QAS is the closest existing system to our prefix-aware reordering. Both prioritize requests by KV segment reusability; our approach uses a simpler greedy sort by prefix hit length, while QAS uses a three-counter demand signal. **The critical gap:** PRISM uses static chunk sizes and does not measure, identify, or address the TPOT degradation that prefix-aware scheduling introduces. Our work begins where PRISM stops — we show that QAS-style scheduling induces a specific starvation failure mode, and that a feedback controller on decode queue depth is necessary to make prefix-aware scheduling safe.

**FEATHER (arXiv:2605.06046, 2026)**  
RL-based scheduler that groups prefix-homogeneous requests into the same batch using a learned policy. Reports 2–10× throughput improvement. More expressive than greedy sorting but requires training a policy network and is evaluated in a disaggregated architecture.

*Overlap:* FEATHER and our reordering share the core insight (group similar-prefix requests together). Our approach argues that a greedy per-step sort achieves comparable TTFT gains at single-node scale without the training overhead of RL. FEATHER does not address TPOT degradation or chunk control.

**AlignedServe (arXiv:2605.23389, 2026)**  
Groups requests by similar prefix length to eliminate iteration-level bubbles in disaggregated serving. Focused on batching structure and inter-node alignment rather than per-step queue resorting or KV eviction.

*Overlap:* Low. Targets a disaggregated architecture; does not address the scheduling-starvation-chunk coupling.

**CacheWise (arXiv:2606.16824, 2026)**  
Combines prefix-aware scheduling with reuse-aware eviction for coding agent workloads. Covers two of our three layers (scheduling + eviction) but has no adaptive chunk controller and does not identify decode starvation as a scheduling side effect.

*Overlap:* Medium. Closest to our eviction + scheduling combination but domain-specific (coding agents) and missing the third layer. Does not report TPOT.

---

## 3. KV Cache Eviction

**Recency/Frequency Adaptive KV Caching (arXiv:2606.21238, 2026)**  
Proposes a hybrid LRU/LFU eviction policy that dynamically balances recency and frequency scores. Reports +10.8% KV cache hit rate and −12.6% TTFT over vanilla vLLM.

*Overlap:* Direct overlap with our CF eviction policy. **We adopt this as a system component and cite it as prior work rather than claiming it as a contribution.**

---

## 4. Overlap Analysis

| Dimension | Our work | PRISM | SlidingServe | FEATHER | CacheWise |
|-----------|----------|-------|-------------|---------|-----------|
| Prefix-aware reordering | ✓ greedy | ✓ demand-scored | ✗ | ✓ RL | ✓ |
| Adaptive chunk control | ✓ queue-depth | ✗ static | ✓ SLO-slack | ✗ | ✗ |
| Reordering → starvation identified | ✓ | ✗ | ✗ | ✗ | ✗ |
| Queue-depth as feedback signal | ✓ | ✗ | ✗ | ✗ | ✗ |
| Coupling between scheduling + chunk | ✓ | ✗ | ✗ | ✗ | ✗ |
| TPOT reported | ✓ | ✗ | SLO-framed | ✗ | ✗ |
| KV eviction co-design | ✓ (adopted) | ✓ DART | ✗ | ✗ | ✓ |

**Key gap in all prior work:** No existing paper identifies that prefix-aware scheduling induces a specific decode starvation pattern, nor that a queue-depth-driven chunk controller is the mechanism to correct it. PRISM covers the scheduling half with static chunks and ignores TPOT. SlidingServe covers the chunk-control half with deadline signals and ignores prefix caching. Neither paper identifies the causal connection between the two.

---

## 5. Positioning Statement

*(This is the core argument for the Related Work section and the Introduction.)*

PRISM demonstrates that prefix-aware scheduling reduces TTFT by 23–37% but does not measure the TPOT degradation it introduces, and uses a static chunk size that cannot respond to scheduling-induced decode pressure. SlidingServe demonstrates that adaptive chunk sizing can protect decode throughput, but requires per-request SLO deadlines as input and is entirely decoupled from prefix caching. We identify the missing connection between these two lines of work: prefix-aware request reordering causes a specific starvation pattern in the decode queue, detectable without per-request deadlines from queue depth alone, and a queue-driven chunk controller combined with prefix-aware scheduling produces super-additive gains that neither system achieves alone.

The key claim is not that any single component is novel — prefix-aware scheduling (PRISM, FEATHER) and adaptive chunking (SlidingServe) both have prior art — but that the **coupling** between them has not been identified, measured, or exploited. Our contribution is the causal chain:

> prefix-aware reordering → decode starvation → queue-depth controller → TPOT protection → super-additive TTFT + TPOT improvement

and the experimental confirmation that the combined system is Pareto-dominant over any single-intervention baseline on every tracked metric.

---

## 6. What to Verify Before Submission

- [ ] Read FEATHER (arXiv:2605.06046) in full — confirm it does not identify TPOT degradation from grouping
- [ ] Read CacheWise (arXiv:2606.16824) in full — confirm no chunk controller or starvation analysis
- [ ] Check whether PRISM's cold lane (FIFO starvation protection) implicitly solves the TPOT problem — if yes, our chunk controller's novelty narrows
- [ ] Search for any paper combining prefix-aware scheduling with queue-depth-driven chunk control published after June 2026
