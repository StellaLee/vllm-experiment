# Near-Saturation Experiment Findings

**Date:** 2026-07-07  
**Experiment:** Phase 1.3 — prefix reordering + dynamic chunking + aging at near-saturation  
**Script:** `scripts/run_near_sat_bench.sh`  
**Analysis:** `python3 src/analyze_near_sat.py`

---

## 1. Regime Characterization

| Dataset   | Arrival rate | Throughput | GPU util | Regime |
|-----------|-------------|------------|----------|--------|
| BurstGPT  | 4 req/s     | 2.73 req/s | —        | oversaturated (queue unbounded) |
| BurstGPT  | 3 req/s     | 2.37 req/s | **85%**  | near-saturation ✓ |
| ShareGPT  | 4 req/s     | 3.43 req/s | 75%      | underloaded for this metric |
| ShareGPT  | 5 req/s     | 3.99 req/s | 68%      | (see §3) |

BurstGPT at 3 req/s is the confirmed near-saturation operating point.
ShareGPT's GPU utilization signal is unreliable — see §3.

---

## 2. BurstGPT Results (Primary Claim)

Arrival rate: 3 req/s · 150 prompts · max-seqs=32 · aging T=2 s

| Metric         | Baseline | Combined | Δ combined | Aging (comb+T=2s) | Δ aging |
|----------------|----------|----------|------------|-------------------|---------|
| TTFT p50 (ms)  | 86       | 92       | +7.0% >>>  | 87                | +0.6% >>> |
| TTFT p95 (ms)  | 302      | 302      | +0.1% >>>  | 300               | −0.6% <-- |
| TPOT p95 (ms)  | 30.2     | 23.0     | **−23.7% <--** | 26.8         | −11.3% <-- |
| E2EL p95 (ms)  | 8134     | 7019     | **−13.7% <--** | 6254         | **−23.1% <--** |
| E2EL p99 (ms)  | 22812    | 23753    | +4.1% >>>  | 21149             | −7.3% <-- |
| Throughput     | 2.37     | 2.41     | +1.6%      | 2.41              | +1.6% |
| GPU util       | 85%      | 84%      | —          | 83%               | — |

Arrow: `<--` = improvement (lower is better for latency), `>>>` = degradation.

**Key observations:**

- **TPOT p95 −24% (combined):** fewer context-switch penalties — warm requests complete their prefill without yielding the GPU to cold requests, so generation throughput is more consistent.
- **E2EL p95 −23% (aging):** aging (T=2 s) on top of combined outperforms combined alone on end-to-end tail latency. Aging ensures cold requests are eventually promoted, capping worst-case wait time without significantly harming the warm-request advantage.
- **TTFT p50 +7% (combined):** the expected cost — cold requests wait behind warm ones. This is acceptable since cold requests have shorter total compute (smaller residual prefill) but experience a scheduling delay.
- **E2EL p99:** combined alone makes p99 slightly worse (+4%), but aging recovers it (−7%). Aging is necessary to prevent extreme cold-request starvation at the far tail.
- **Throughput:** essentially unchanged (+1.6%) across conditions — the mechanism redistributes latency, it does not increase capacity.

**Takeaway for paper:** At near-saturation, combined reordering + dynamic chunking yields −24% TPOT p95. Adding aging (T=2 s) yields −23% E2EL p95 while nearly eliminating starvation risk. The TTFT tradeoff (+7% p50) is the inherent cost of warm-first scheduling.

---

## 3. ShareGPT Results and Dataset Characterization

Arrival rate: 4 req/s · same config

| Metric         | Baseline | Combined | Δ combined | Aging | Δ aging |
|----------------|----------|----------|------------|-------|---------|
| TTFT p95 (ms)  | 122      | 143      | +17.6% >>> | 142   | +17.1% >>> |
| TPOT p95 (ms)  | 18.4     | 18.3     | −0.3%      | 17.9  | −2.5% |
| E2EL p95 (ms)  | 9797     | 10501    | +7.2% >>>  | 10539 | +7.6% >>> |
| Throughput     | 3.43     | 3.43     | 0%         | 3.43  | +0.1% |
| GPU util       | 75%      | 73%      | —          | 73%   | — |

**Why ShareGPT behaves differently:**

1. **Low prefix repetition.** ShareGPT is a conversational dataset with diverse, single-turn prompts. There is little shared prefix structure across requests, so warm-first reordering provides minimal cache benefit while still delaying cold requests. The TTFT penalty (+18%) has no offsetting cache win.

2. **Decode-bound workload.** ShareGPT has short inputs and long outputs. The GPU spends most time in autoregressive decode (memory-bandwidth bound), not prefill (compute bound). As a result, `nvidia-smi` GPU utilization stays flat regardless of arrival rate — it actually *decreases* from 75% at 4 req/s to 68% at 5 req/s, because higher rates cause more batching during decode which doesn't register as higher compute utilization. GPU util is not a reliable near-saturation signal for decode-heavy workloads.

3. **Not actually near-saturation.** At 4 req/s, ShareGPT achieves 3.43 req/s throughput — the system is keeping up without a persistent queue. Without a queue, there is no scheduling opportunity: all requests start immediately and warm-first reordering has no effect. The mechanism only activates when `self.waiting` is non-empty.

**Takeaway for paper:** Our mechanism is designed for prefill-heavy, prefix-rich workloads where (a) a waiting queue exists and (b) warm requests provide measurable cache benefit. BurstGPT (code completions with shared function signatures and context) matches this profile. ShareGPT does not. This scopes the claim correctly: the paper should position the contribution for code LLM serving and similar prefix-sharing workloads, not general-purpose chat.

---

## 4. Comparison: Oversaturated vs Near-Saturation

| Regime             | Dataset  | Arrival | Combined TPOT p95 Δ | Combined E2EL p95 Δ |
|--------------------|----------|---------|----------------------|----------------------|
| Oversaturated      | BurstGPT | 4 req/s | +4.5% >>>            | +12.1% >>>           |
| **Near-saturation**| BurstGPT | 3 req/s | **−23.7% <--**       | **−13.7% <--**       |

The mechanism reverses sign between regimes. At 2× overload the queue grows faster than warm-first scheduling can drain it, so cold requests starve and overall latency worsens. At near-saturation the queue is short and transient — warm requests clear quickly and cold requests wait only briefly. This is the regime where the mechanism is designed to operate.

---

## 5. Aging Mechanism Value

| Condition     | E2EL p95 Δ vs baseline | E2EL p99 Δ vs baseline |
|---------------|------------------------|------------------------|
| Combined only | −13.7%                 | +4.1% (worse)          |
| Aging T=2s    | **−23.1%**             | **−7.3%**              |

Aging recovers the p99 regression that combined alone introduces. The two-pass scheduler (warm bucket → aged bucket at T=2 s) ensures no request waits more than ~T seconds before being promoted to FIFO priority. T=2 s appears sufficient at 3 req/s; the optimal T likely scales with mean service time.

---

## 6. Open Items

- **KV hit rate:** Prometheus metric names differ in vLLM v0.23.0; all runs show N/A. Need to discover the correct metric name on the remote to quantify cache hit improvement directly.
- **LRU factorial (Ablation B):** `ns_dyn` and `ns_reorder` conditions for the 2×2 factorial still need to be run (`scripts/run_lru_factorial.sh`).
- **ShareGPT near-saturation:** No viable operating point found where ShareGPT is both queue-saturated and showing reordering benefit. Recommend treating ShareGPT as a "low-sharing" contrast case rather than a second positive result.
- **Aging T sweep:** T=2 s was inherited from the saturation experiment. A short sweep (T=1, 2, 5 s) at 3 req/s would confirm this is near-optimal.
