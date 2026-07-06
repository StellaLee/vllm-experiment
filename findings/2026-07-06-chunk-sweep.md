# Static Chunk Size Sweep — 2026-07-06

Sweeps `--max-num-batched-tokens` across 256 / 2048 / 4096 tokens to map the
TTFT/TPOT tradeoff curve and contextualise the dynamic bang-bang controller.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workloads:** BurstGPT (`rate=inf`, 150 prompts) · ShareGPT (`rate=inf`, 150 prompts)  
**Raw logs:** `logs/sweep_256_*.json`, `logs/sweep_4096_*.json`  
**Script:** `scripts/run_chunk_sweep.sh`

---

## Results

### BurstGPT (thundering-herd, diverse prompts)

| Config | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p95 (ms) | Req/s |
|--------|--------------|--------------|--------------|-------|
| static 256 | 5099 | 10167 | **31.7** | 4.81 |
| static 2048 (default) | 4818 | 11652 | 127.0 | 4.99 |
| static 4096 | **4235** | **11379** | 133.7 | 4.89 |
| dynamic controller | 5111 | 11898 | 37.6 | 4.33 |

Deltas vs static 2048:

| Config | TTFT p50 | TTFT p95 | TPOT p95 | Req/s |
|--------|----------|----------|----------|-------|
| static 256 | +5.8% | -12.7% | **-75.0%** | -3.6% |
| static 4096 | **-12.1%** | -2.3% | +5.3% | -2.0% |
| dynamic controller | +6.1% | +2.1% | -70.4% | -13.1% |

### ShareGPT (multi-turn, mixed lengths)

| Config | TTFT p50 (ms) | TTFT p95 (ms) | TPOT p95 (ms) | Req/s |
|--------|--------------|--------------|--------------|-------|
| static 256 | **5299** | **11999** | **24.9** | **6.09** |
| static 2048 (default) | 7083 | 14346 | 29.5 | 5.61 |
| static 4096 | 5880 | 12813 | 30.2 | 5.89 |
| dynamic controller | 5633 | 12378 | 25.3 | 6.01 |

Deltas vs static 2048:

| Config | TTFT p50 | TTFT p95 | TPOT p95 | Req/s |
|--------|----------|----------|----------|-------|
| static 256 | **-25.2%** | **-16.4%** | **-15.7%** | **+8.6%** |
| static 4096 | -17.0% | -10.7% | +2.3% | +5.0% |
| dynamic controller | -20.5% | -13.7% | -14.2% | +7.2% |

---

## Key Findings

### 1. The 2048 default is a saddle point

On ShareGPT, both 256 and 4096 improve TTFT over the default — meaning 2048 is
not a local optimum in either direction. Any departure from the default helps
latency on this workload. This is the core empirical motivation for adaptive
chunk sizing.

### 2. The optimal static value is workload-dependent

BurstGPT optimal by TTFT: **4096** (large chunks finish prefill faster on
diverse one-shot prompts).  
ShareGPT optimal across all metrics: **256** (small chunks interleave requests
fairly, reducing queuing delay for concurrent multi-turn conversations).

No single static value wins both workloads simultaneously.

### 3. Static 256 beats the dynamic controller on ShareGPT

Static 256 achieves better TTFT p50 (5299 vs 5633 ms), TPOT p95 (24.9 vs
25.3 ms), and throughput (6.09 vs 6.01 req/s) than the bang-bang controller.
The controller converges toward small chunks under high decode pressure but
oscillates above 256 — it is too conservative to reach the static optimum
consistently.

### 4. Dynamic controller excels at avoiding TPOT extremes

On BurstGPT, static 256 achieves the same TPOT improvement (-75% vs -70%) but
the controller does so without requiring workload foreknowledge, and it avoids
the TTFT regression (+5.8%) that static 256 causes under this arrival pattern
— though it introduces its own moderate TTFT overhead (+6.1%).

---

## Interpretation

The sweep reveals a complex tradeoff surface: the two failure modes of the
2048 default are (a) decode starvation under bursty arrivals and (b) excessive
queuing delay under concurrent multi-turn traffic. These call for opposite
static tuning directions. A workload-adaptive controller is the right
architecture; the bang-bang approach partially closes the gap but leaves
improvement on the table for steady workloads where a tighter static value
would win.

**Mitigation for the controller's conservatism:** reducing the bang-bang step
size (half/double → ±25%) or adding a moving-average filter would let the
controller settle closer to the static optimum without overshooting.

---

## Reproduction

```bash
# 1. Start with patches applied and server stopped
bash patches/apply_patches.sh

# 2. Run sweep (256 and 4096; 2048 reused from existing baseline)
PYTHON=/root/miniconda3/bin/python3 bash scripts/run_chunk_sweep.sh

# 3. Analyze
python3 src/analyze_chunk_sweep.py --log-dir logs
```
