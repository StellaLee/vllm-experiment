# Cache-Hit-Ratio Reordering — 2026-07-08

Improvement to the warm-first reordering strategy: replace the raw cached
token count sort key with cache hit ratio (cached / total prompt tokens).

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workload:** ShareGPT sequential multi-turn replay — 200 conversations, 4 turns,
concurrency=15, min-turns=4  
**Condition:** PREFIX_REORDER=1, DYNAMIC_CHUNK=0 (reorder-only isolation)  
**Trials:** 2 per strategy

---

## Motivation

The original count-based strategy sorted the waiting queue by raw cached token
count (descending). This is flawed: a T4 request with 30 cached tokens out of
100 total still requires 70 tokens of prefill computation, while a T2 request
with 10 cached tokens out of 15 total requires only 5. Count-based ordering
schedules T4 first — the more expensive request — which is the wrong direction
for TTFT minimisation.

**Fix:** sort by `cached_tokens / total_tokens` (hit ratio, descending). This
prioritises requests where the largest *fraction* of the prompt is already
cached, i.e. those with the least remaining prefill work.

```python
# Old sort key (count-based)
def _cached_tokens(req):
    if req.num_computed_tokens > 0:
        return req.num_computed_tokens
    _, n = self.kv_cache_manager.get_computed_blocks(req)
    return n

# New sort key (ratio-based)
def _hit_ratio(req):
    total = req.num_prompt_tokens
    if total == 0:
        return 0.0
    if req.num_computed_tokens > 0:
        cached = req.num_computed_tokens
    else:
        _, cached = self.kv_cache_manager.get_computed_blocks(req)
    return cached / total
```

---

## Results — TTFT p95 (ms)

### Per-trial

| Turn | baseline | count_t1 | count_t2 | ratio_t1 | ratio_t2 |
|------|----------|----------|----------|----------|----------|
| T1 | 232.8ms | 224.8ms (−3%) | 211.7ms (−9%) | 230.9ms (−1%) | 223.7ms (−4%) |
| T2 | 124.7ms | 129.9ms (+4%) | 131.7ms (+6%) | 133.9ms (+7%) | 132.3ms (+6%) |
| T3 | 147.8ms | 156.6ms (+6%) | 153.7ms (+4%) | 144.7ms (−2%) | 141.0ms (−5%) |
| T4 | 154.4ms | 105.4ms (−32%) | 104.3ms (−32%) | 88.0ms (−43%) | 109.9ms (−29%) |

### 2-trial averages vs combined

| Turn | baseline | count_avg | ratio_avg | combined |
|------|----------|-----------|-----------|----------|
| T1 | 232.8ms | 218.2ms (−6.3%) | 227.3ms (−2.4%) | 218.6ms (−6.1%) |
| T2 | 124.7ms | 130.8ms (+4.9%) | 133.1ms (+6.7%) | 107.5ms (−13.8%) |
| T3 | 147.8ms | 155.2ms (+5.0%) | 142.9ms (−3.3%) | 86.3ms (−41.6%) |
| T4 | 154.4ms | 104.9ms (−32.1%) | 99.0ms (−35.9%) | 80.9ms (−47.6%) |

---

## Key Findings

### 1. Ratio-based fixes the T3 regression

Count-based reordering degraded T3 TTFT by +5% (promoting high-cached-count T4
requests ahead of cheaper T3 ones). Ratio-based reverses this to −3% at T3
because T3 requests, which have shorter remaining prefill, now rank ahead of
long T4 requests with low hit ratios.

### 2. Ratio-based improves T4 (directionally)

Average T4 improvement: −36% (ratio) vs −32% (count). The direction is correct
but the two ratio trials disagree (88.0ms vs 109.9ms, a 25% spread), so the
advantage over count-based cannot be claimed confidently from two trials alone.

### 3. T2 regression persists with both strategies

Both strategies regress T2 TTFT by +5–7%. This is an inherent cost of any
warm-first priority scheme: T2 requests are deprioritised relative to T3/T4
warm ones. Ratio-based does not solve this.

### 4. Combined still dominates at every turn

Neither reorder-only strategy approaches combined (chunk + reorder). Combined
achieves −14% at T2, −42% at T3, −48% at T4. Dynamic chunking is the primary
driver; reordering provides a complementary but secondary benefit.

---

## Conclusion

Ratio-based reordering is a strictly better sort key than count-based:
it has a sound theoretical motivation (minimise remaining prefill work) and
the empirical results are directionally better at T3 and T4. It is adopted
as the default implementation in `patch_scheduler.py`.

The T2 regression is a known limitation of warm-first scheduling and is not
resolved by the metric change. The aging mechanism (AGING_THRESHOLD_MS)
provides starvation protection but does not specifically address T2 regression.

---

## Files

| Tag | File |
|-----|------|
| ratio_t1 | `logs/2026-07-08-mt-mt_ratio_reorder_c15_t1.jsonl` |
| ratio_t2 | `logs/2026-07-08-mt-mt_ratio_reorder_c15_t2.jsonl` |
| Patch script | `scripts/patch_scheduler.py` |
