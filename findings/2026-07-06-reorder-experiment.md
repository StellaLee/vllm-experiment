# Prefix-Aware Request Reordering — 2026-07-06

Tests whether sorting the waiting queue by cached prefix length before each
scheduling step reduces TTFT by prioritising cache-warm requests.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workloads:** BurstGPT (`rate=inf`, 150 prompts) · ShareGPT (`rate=inf`, 150 prompts)  
**Patch:** `patches/reorder/scheduler.patch` — enabled via `PREFIX_REORDER=1`  
**Script:** `scripts/run_reorder_experiment.sh`

---

## Mechanism

At the start of each scheduling step, before the FCFS waiting loop runs,
the waiting queue is re-sorted in descending order of cached prefix token
count:

```python
_sorted = sorted(self.waiting, key=_cached_tokens, reverse=True)
self.waiting.clear()
self.waiting.extend(_sorted)
```

`_cached_tokens` calls `kv_cache_manager.get_computed_blocks(req)` which
does a prefix hash tree lookup — a read-only O(prefix_depth) operation.
Requests with 0 cached tokens (cold) fall to the back; requests whose
prompt prefix is already in the block pool are scheduled first.

---

## Results

### BurstGPT (thundering-herd, diverse prompts)

| Metric | Reorder OFF | Reorder ON | Delta |
|--------|------------|-----------|-------|
| TTFT p50 (ms) | 4440 | 4121 | **-7.2%** |
| TTFT p95 (ms) | 11062 | 9980 | **-9.8%** |
| TTFT p99 (ms) | 11675 | 10423 | **-10.7%** |
| TPOT p50 (ms) | 35.0 | 45.4 | +29.8% |
| TPOT p95 (ms) | 112.3 | 116.4 | +3.7% |
| E2EL p95 (ms) | 16247 | 15093 | **-7.1%** |
| Throughput (req/s) | 5.00 | 5.16 | **+3.3%** |

### ShareGPT (multi-turn, shared prefixes)

| Metric | Reorder OFF | Reorder ON | Delta |
|--------|------------|-----------|-------|
| TTFT p50 (ms) | 5375 | **3966** | **-26.2%** |
| TTFT p95 (ms) | 12838 | 12245 | **-4.6%** |
| TTFT p99 (ms) | 14197 | 12978 | **-8.6%** |
| TPOT p50 (ms) | 20.3 | 21.4 | +5.3% |
| TPOT p95 (ms) | 26.4 | 35.5 | +34.7% |
| E2EL p95 (ms) | 19851 | 17765 | **-10.5%** |
| Throughput (req/s) | 5.84 | 6.11 | **+4.5%** |

---

## Key Findings

### 1. Strong TTFT improvement, especially on ShareGPT

ShareGPT p50 TTFT drops 26.2% — the largest single-intervention TTFT gain
across all experiments. BurstGPT also improves despite its more diverse prompt
distribution, suggesting enough shared prefix structure (e.g. system prompts,
repeated query patterns) to benefit from reordering.

### 2. TPOT degrades — the prefill/decode tradeoff

Reordering aggressively prioritises cache-warm prefill, which delays cold
requests and allows the decode queue to grow unchecked. BurstGPT TPOT p50
rises 29.8%; ShareGPT TPOT p95 rises 34.7%. This is the direct cost of
one-sided scheduling.

### 3. Reordering and dynamic chunk control are complementary

The TPOT degradation is precisely the failure mode that the dynamic chunk
controller is designed to address: a growing decode queue triggers budget
shrinkage, protecting per-token latency. Running both together should capture
the TTFT gain while mitigating the TPOT penalty — the combined condition is
the natural next experiment.

### 4. Throughput improves regardless

Both workloads show +3–5% throughput improvement. Serving cache-warm requests
first reduces effective compute per token (fewer KV blocks to recompute),
increasing overall serving efficiency.

---

## Interpretation

Prefix-aware reordering reveals that **request ordering is a first-order
lever for TTFT** — as impactful as eviction policy changes, and orthogonal to
them. The TPOT cost is a predictable consequence of unguarded prefill
prioritisation; the dynamic chunk controller provides the natural guard.

The three prefill-side interventions now form a coherent hierarchy:

1. **CF eviction** — retain the right blocks under pressure
2. **Prefix reordering** — schedule cache-warm requests first
3. **Dynamic chunk control** — prevent decode starvation when prefill surges

---

## Reproduction

```bash
# 1. Apply the reordering patch
python3 patches/reorder/apply_patch.py

# 2. Run baseline vs reordering on both datasets
PYTHON=/root/miniconda3/bin/python3 bash scripts/run_reorder_experiment.sh

# 3. Analyze
python3 src/analyze_reorder.py --log-dir logs
```

### Env vars

| Variable | Default | Description |
|----------|---------|-------------|
| PREFIX_REORDER | 0 | Set to 1 to enable |
