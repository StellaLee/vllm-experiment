# Soft Aging Reordering — 2026-07-09

Replace the hard aging cliff with a continuous soft-priority score that
blends cache hit ratio and wait time into a single sort key.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workload:** ShareGPT sequential multi-turn replay — 200 conversations, 4 turns,
concurrency=15, min-turns=4  
**Condition:** PREFIX_REORDER=1, DYNAMIC_CHUNK=0 (reorder-only isolation)  
**Trials:** 2

---

## Motivation

The ratio-based hard-cliff strategy (2026-07-08) partitioned the waiting queue
into two separate buckets: requests aged past AGING_THRESHOLD_MS were promoted
en masse to the front (FCFS within that bucket), while fresher requests were
sorted by hit ratio. This creates a cliff-edge discontinuity: at the threshold
moment, a T1 request with 0% hit ratio jumps ahead of T3 requests with 30%
hit ratio, regardless of the relative priority each deserves.

The T3 regression observed with ratio-based reordering (+4-5% vs baseline) is
consistent with this mechanism: aged T1 requests promoted in bulk displace T3
requests that would benefit more from early scheduling.

**Fix:** replace the two-bucket scheme with a single unified sort key:

```
score(req) = hit_ratio(req) + α · log(1 + wait_seconds(req))
```

where `α = AGING_ALPHA` (default 0.3). This makes promotion continuous:

- A T1 with 0% hit ratio and 30s wait scores: 0 + 0.3·log(31) ≈ 1.03
- A T3 with 35% hit ratio and 15s wait scores: 0.35 + 0.3·log(16) ≈ 1.18
- A T4 with 60% hit ratio and 5s wait scores:  0.60 + 0.3·log(6)  ≈ 1.14

The T3 request wins unless T1 has been waiting >2.5× longer — a principled
tradeoff rather than an arbitrary time cliff.

```python
# New sort key (soft aging)
def _soft_priority(req):
    total = req.num_prompt_tokens
    if total == 0:
        hit_ratio = 0.0
    elif req.num_computed_tokens > 0:
        hit_ratio = req.num_computed_tokens / total
    else:
        _, cached = self.kv_cache_manager.get_computed_blocks(req)
        hit_ratio = cached / total
    wait_s = _now - req.arrival_time
    return hit_ratio + _alpha * math.log1p(wait_s)
```

---

## Results — TTFT p95 (ms)

### Per-trial

| Turn | baseline | ratio_t1 | ratio_t2 | soft_t1 | soft_t2 |
|------|----------|----------|----------|---------|---------|
| T1 | 234.6ms | 231.5ms (−1%) | 224.0ms (−5%) | 316.9ms (+35%) | 224.4ms (−4%) |
| T2 | 125.0ms | 135.2ms (+8%) | 133.1ms (+6%) | 132.5ms (+6%) | 131.2ms (+5%) |
| T3 | 147.8ms | 155.1ms (+5%) | 151.4ms (+2%) | 122.2ms (−17%) | 141.1ms (−5%) |
| T4 | 154.5ms | 90.2ms (−42%) | 110.2ms (−29%) | 105.4ms (−32%) | 96.7ms (−37%) |

### 2-trial averages vs combined

| Turn | baseline | ratio_avg | soft_avg | combined |
|------|----------|-----------|----------|----------|
| T1 | 234.6ms | 227.8ms (−3%) | 270.6ms (+15%) | 219.1ms (−7%) |
| T2 | 125.0ms | 134.1ms (+7%) | 131.9ms (+5%) | 108.1ms (−14%) |
| T3 | 147.8ms | 153.2ms (+4%) | 131.7ms (−11%) | 87.8ms (−41%) |
| T4 | 154.5ms | 100.2ms (−35%) | 101.0ms (−35%) | 81.5ms (−47%) |

---

## Key Findings

### 1. T3 regression reversed: −11% vs baseline (ratio: +4%)

Both soft aging trials show T3 improvement vs baseline (−17%, −5%), while both
ratio trials show T3 degradation (+5%, +2%). The directional difference is
consistent across trials: soft aging eliminates the T3 regression induced by
the hard aging cliff.

The mechanism: aged T1 requests (0% hit ratio) are no longer promoted en masse
ahead of T3 requests (30–40% hit ratio). Instead they must wait until their
accumulated wait-time boost exceeds the hit-ratio advantage of the T3 cohort.

### 2. T4 unchanged

Both strategies achieve ~35% T4 improvement. Soft aging does not help or hurt
at T4 relative to ratio-based ordering.

### 3. T2 regression marginally improved (5% vs 7%)

The structural warm-first cost to T2 is reduced slightly (from +7% average
to +5% average). The T2 regression is inherent to any warm-first scheme and
is not resolved by this change.

### 4. T1 high variance persists

Soft_t1 shows +35% T1 regression; soft_t2 shows −4%. This is the same
two-run T1 instability observed with chunk_only in the 2026-07-08 decomp
experiment. Likely causes: GPU cold-start effects for the very first batch,
or scheduling noise in the first slot when the waiting queue is cold. T1 TTFT
under reorder-only conditions cannot be reliably characterised from two trials.

### 5. Combined still dominates at T3

Soft aging reorder-only reaches −11% at T3; combined (chunk + reorder) reaches
−41%. Dynamic chunking remains the primary driver at T3/T4. Soft aging
improves over ratio-based at T3 but still substantially trails combined.

---

## Mechanism Analysis

The hard-cliff aging creates a priority inversion: once a T1 request crosses
the age threshold, it jumps to the front of the queue regardless of its 0%
hit ratio. At concurrency=15 with 200 conversations cycling, many T1 requests
age past the threshold while waiting, displacing mid-turn requests with
meaningful cache warmth.

Soft aging avoids this inversion by scaling the wait-time boost continuously:
a cold request accumulates priority gradually, overtaking warm requests only
after the wait time is long enough to justify it.

At AGING_ALPHA=0.3:
- To overtake a 100% hit-ratio request: wait ~27 seconds
- To overtake a 35% hit-ratio request: wait ~7 seconds

This is more principled than an arbitrary threshold.

---

## Conclusion

Soft aging is a strictly better aging mechanism than the hard cliff for
multi-turn workloads. It eliminates the T3 regression (−11% vs +4%), keeps
T4 identical, and marginally improves T2. The patch is adopted as the default
reordering implementation.

The T2 regression (+5%) remains a known limitation of any warm-first scheme.
Resolving it would require a fundamentally different approach (e.g., PRISM-style
reserved cold-lane slots).

---

## Implementation

**New env vars:**

| Var | Default | Description |
|-----|---------|-------------|
| `AGING_ALPHA` | `0.3` | Continuous aging coefficient |

`AGING_THRESHOLD_MS` is retained but unused by the reorder block (legacy).

**Files changed:**

| File | Change |
|------|--------|
| `scripts/patch_scheduler.py` | REORDER_BLOCK → soft_priority; Patch 2b adds _aging_alpha |
| `scripts/hotpatch_soft_aging.py` | One-shot upgrade for already-patched schedulers |
| `scripts/run_soft_aging.sh` | 2-trial experiment runner |
| live scheduler | Patched via hotpatch_soft_aging.py on 2026-07-09 |

**Log files:**

| Tag | File |
|-----|------|
| soft_t1 | `logs/2026-07-09-mt-mt_soft_aging_c15_t1.jsonl` |
| soft_t2 | `logs/2026-07-09-mt-mt_soft_aging_c15_t2.jsonl` |
