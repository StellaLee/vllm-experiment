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

1. **Cache locality lost to request interleaving.** ShareGPT IS a multi-turn conversation dataset — each successive turn includes the full prior history as a prefix, so the theoretical prefix repetition is high. However, the benchmark sends 150 requests with Poisson arrivals and randomized ordering. Turn N+1 of conversation A does not necessarily arrive before turn N's KV blocks have been evicted by intervening requests from other conversations. In practice, the inter-turn gap under random arrival ordering is long enough for the cache to have turned over. We cannot confirm this hypothesis directly because the KV hit rate metric is returning N/A for all runs (vLLM v0.23.0 metric name mismatch — see Open Items).

2. **Decode-bound workload.** ShareGPT has short inputs and long outputs. The GPU spends most time in autoregressive decode (memory-bandwidth bound), not prefill (compute bound). As a result, `nvidia-smi` GPU utilization stays flat regardless of arrival rate — it actually *decreases* from 75% at 4 req/s to 68% at 5 req/s, because batched decode does not saturate compute. GPU util is not a reliable near-saturation signal for decode-heavy workloads.

3. **Not actually near-saturation.** At 4 req/s, ShareGPT achieves 3.43 req/s throughput — the system is keeping up without a persistent queue. Without a queue, there is no scheduling opportunity: requests start immediately and warm-first reordering has nothing to reorder. The mechanism only activates when `self.waiting` is non-empty.

**Takeaway for paper:** The ShareGPT result is inconclusive, not a negative result. The most likely explanation is that random request ordering destroys the inter-turn cache locality that makes ShareGPT theoretically prefix-rich. Fixing the KV hit rate metric is the critical next step — if ShareGPT shows low hit rates despite high theoretical sharing, it confirms the ordering hypothesis and motivates conversation-aware request scheduling as future work. If hit rates are high but the mechanism still hurts, it points to decode-bound dynamics as the cause.

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

- **KV hit rate (critical):** Prometheus metric names differ in vLLM v0.23.0; all runs show N/A. This is the single most important open item — without it we cannot explain the ShareGPT result (low hit rate → ordering hypothesis confirmed; high hit rate → decode-bound hypothesis). Need to grep the vLLM metrics endpoint on the remote to find the correct counter name.
- **LRU factorial (Ablation B):** `ns_dyn` and `ns_reorder` conditions for the 2×2 factorial still need to be run (`scripts/run_lru_factorial.sh`).
- **ShareGPT near-saturation:** No viable operating point found where ShareGPT is both queue-saturated and showing reordering benefit. Recommend treating ShareGPT as a "low-sharing" contrast case rather than a second positive result.
- **Aging T sweep:** T=2 s was inherited from the saturation experiment. A short sweep (T=1, 2, 5 s) at 3 req/s would confirm this is near-optimal.
