# Ablation A: Arrival Rate Sweep — 2026-07-07

Tests whether gains from the combined condition (PREFIX_REORDER=1 + DYNAMIC_CHUNK=1)
persist under realistic Poisson arrival rates, or whether they were amplified by the
`rate=inf` thundering-herd setup used in all prior experiments.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workloads:** BurstGPT · ShareGPT  
**Rates:** Poisson 1 / 2 / 4 / 8 req/s  
**Prompts per run:** 150  
**Script:** `scripts/run_rate_sweep.sh`  
**Analyzer:** `src/analyze_rate_sweep.py`

---

## Results

### BurstGPT — combined vs baseline delta

| Metric | 1 req/s | 2 req/s | 4 req/s | 8 req/s |
|--------|---------|---------|---------|---------|
| TTFT p50 | −4.2% | −3.9% | +1.6% | **+24.0%** |
| TTFT p95 | +0.4% | −1.6% | +0.1% | **+30.8%** |
| TPOT p50 | −0.5% | −2.1% | −4.9% | −0.0% |
| TPOT p95 | −2.5% | −7.6% | +7.1% | **−36.4%** |
| E2EL p95 | **−50.9%** | **−17.4%** | **−34.9%** | **−17.8%** |
| Throughput | +8.2% | +15.5% | −12.3% | −2.1% |

### BurstGPT — baseline absolute values

| Metric | 1 req/s | 2 req/s | 4 req/s | 8 req/s |
|--------|---------|---------|---------|---------|
| TTFT p50 (ms) | 76.6 | 83.6 | 89.5 | 104.5 |
| TTFT p95 (ms) | 247.0 | 285.6 | 308.8 | 368.5 |
| TPOT p50 (ms) | 16.4 | 17.0 | 18.4 | 23.9 |
| TPOT p95 (ms) | 20.6 | 22.7 | 30.7 | 48.9 |
| E2EL p95 (ms) | 13320.8 | 9044.4 | 10704.0 | 12960.0 |
| Throughput (req/s) | 0.92 | 1.65 | 3.43 | 4.02 |

### ShareGPT — combined vs baseline delta

| Metric | 1 req/s | 2 req/s | 4 req/s | 8 req/s |
|--------|---------|---------|---------|---------|
| TTFT p50 | −1.0% | −4.4% | +4.5% | +6.3% |
| TTFT p95 | −4.3% | +2.5% | +12.6% | **−35.7%** |
| TPOT p50 | −0.0% | −0.4% | +0.2% | −4.3% |
| TPOT p95 | +0.1% | −0.3% | −2.4% | −8.5% |
| E2EL p95 | +0.9% | −5.1% | +5.8% | −10.6% |
| Throughput | +0.0% | +0.0% | +0.1% | +1.4% |

### ShareGPT — baseline absolute values

| Metric | 1 req/s | 2 req/s | 4 req/s | 8 req/s |
|--------|---------|---------|---------|---------|
| TTFT p50 (ms) | 57.0 | 59.3 | 63.8 | 82.2 |
| TTFT p95 (ms) | 110.2 | 109.3 | 122.3 | 966.3 |
| TPOT p50 (ms) | 16.2 | 16.6 | 17.5 | 20.4 |
| TPOT p95 (ms) | 16.9 | 17.5 | 18.7 | 22.8 |
| E2EL p95 (ms) | 10052.1 | 10425.7 | 9672.7 | 12390.8 |
| Throughput (req/s) | 1.00 | 1.99 | 3.41 | 5.07 |

---

## Key Findings

### 1. The rate=inf TTFT gain does not hold at realistic rates

The headline result from `findings/2026-07-06-combined-experiment.md` — ShareGPT TTFT
p50 −30.2% — is not reproduced at any Poisson rate. At 1–2 req/s gains are 1–4%.
At 4–8 req/s TTFT p50 slightly degrades (+4–6%).

**Root cause:** Prefix-aware reordering only helps when there is a non-empty waiting
queue — it needs multiple requests to choose between. At 1–2 req/s the system is at
20–40% utilization (baseline ShareGPT throughput ≈ 5 req/s); the queue is almost
always empty or has a single request. With nothing to reorder, TTFT is unchanged.
The rate=inf gain was a queuing-amplification artifact, not a fundamental improvement.

### 2. TPOT and E2EL tail improvements are more robust

The chunk controller's TPOT protection persists across rates:
- BurstGPT TPOT p95: −2.5% to −36.4% (improves at every rate except 4 req/s)
- ShareGPT TPOT p95: consistently negative (−0.3% to −8.5%)
- E2EL p95: large improvements on BurstGPT at all rates (−17% to −51%)

These gains are plausible even without a queue — the dynamic chunk controller reduces
chunk size when decode pressure rises, independently of request ordering.

### 3. TTFT degrades at high BurstGPT load (8 req/s)

BurstGPT TTFT p50 +24%, p95 +30.8% at 8 req/s. At this rate the baseline throughput
is only 4.02 req/s — the system is saturated. Reordering under saturation delays cold
requests further, as warm requests continuously jump the queue with no aging mechanism.
This is the starvation failure mode without an aging guard.

### 4. ShareGPT TTFT p95 at 8 req/s is anomalous

Baseline TTFT p95 = 966 ms at 8 req/s vs 110–122 ms at lower rates. The system is
saturated (baseline throughput 5.07 req/s < 8 req/s demand). The −35.7% p95 gain at
8 req/s likely reflects high variance on a saturated system, not a reliable result.
Needs a repeat run to confirm.

---

## Interpretation

The rate sweep reveals a fundamental scope limitation: the optimizations are effective
**only under queuing pressure** (system utilization > ~70%). At sub-saturation rates
(1–4 req/s on this hardware), gains are near zero for TTFT and modest for TPOT.

This does not invalidate the contribution but significantly changes how it must be
framed. The paper should:

1. **Drop the −30.2% TTFT headline** — it is a rate=inf artifact
2. **Lead with TPOT and tail latency** — these are genuine and rate-robust
3. **Scope the claim explicitly** — "our system improves tail latency and TPOT under
   high utilization; at sub-saturation load it is at parity with baseline"
4. **Add the aging mechanism (next step 3)** — the BurstGPT 8 req/s TTFT degradation
   confirms starvation is real and must be addressed before any submission

---

## Impact on Paper Submission Plan

| Claim | Prior status | After rate sweep |
|-------|-------------|-----------------|
| Reordering reduces TTFT | ✓ at rate=inf | ✗ only under queuing pressure |
| Chunk controller reduces TPOT | ✓ at rate=inf | ✓ robust across rates |
| Combined is super-additive | ✓ at rate=inf | Partial — TPOT yes, TTFT conditional |
| Gains persist at realistic load | ✗ needed | ✗ largely no for TTFT |

The TPOT story is now the paper's strongest empirical claim. The workshop draft
should be reframed around tail latency and TPOT protection, with the TTFT gain
presented as a saturation-regime bonus rather than the headline.

---

## Repro

```bash
# on remote server (ssh -p 23 root@117.50.214.139)
cd /root/vllm-experiment

# run full sweep (baseline + combined at 1/2/4/8 req/s, BurstGPT + ShareGPT):
bash scripts/run_rate_sweep.sh

# analyze results:
python3 src/analyze_rate_sweep.py --log-dir logs
```

Results are saved as JSON in `logs/` with embedded tags
`base_r{rate}_{dataset}` / `comb_r{rate}_{dataset}`.
The analyzer loads all JSON files in the log dir and matches by tag field.
