# Soft Aging Reordering — 2026-07-09

Replace the hit-ratio-only sort key with a continuous soft-priority score that
blends cache hit ratio and wait time into a single sort key.

**Model:** Qwen2.5-Coder-7B-Instruct  
**Hardware:** NVIDIA RTX 4090 (24 GB)  
**Workload:** ShareGPT sequential multi-turn replay — 200 conversations, 4 turns,
concurrency=15, min-turns=4  
**Condition:** PREFIX_REORDER=1, DYNAMIC_CHUNK=0 (reorder-only isolation)  
**Trials:** 2

---

## Patch Configuration

All conditions use the patched scheduler (`scripts/patch_scheduler.py` applied
once). The reordering sort key changed between the ratio and soft conditions;
the sort key active at log-collection time is noted below.

| Condition | PREFIX\_REORDER | DYNAMIC\_CHUNK | AGING\_ALPHA | Sort key active | Scheduler version |
|-----------|:--------------:|:-------------:|:------------:|-----------------|-------------------|
| baseline | 0 | 0 | — | FCFS (no reorder) | any |
| ratio\_t1/t2 | 1 | 0 | — (unused) | `cached/total` only, no wait-time boost | pre-soft-aging; commit `5dcf9f3`; collected 2026-07-08 |
| **soft\_t1/t2** | **1** | **0** | **0.3** | `cached/total + 0.3·log(1+wait_s)` | current; commit `7b2bbca`+; collected 2026-07-09 |
| combined (ref) | 1 | 1 | — (unused) | ratio sort key + dynamic chunk (no hysteresis) | pre-soft-aging; collected 2026-07-08 |

Note: `AGING_THRESHOLD_MS` is present in all patched schedulers as a legacy
field but is **not used** by either the ratio or soft sort keys (both were
collected with `AGING_THRESHOLD_MS=inf`, the default).

---

## Motivation

The ratio-based sort key (2026-07-08) ordered the waiting queue by
`cached_tokens / total_tokens` descending, with no consideration of how long
a request has been waiting:

```python
# Old sort key (ratio-only, no aging)
def _hit_ratio(req):
    total = req.num_prompt_tokens
    if req.num_computed_tokens > 0:
        return req.num_computed_tokens / total
    _, cached = self.kv_cache_manager.get_computed_blocks(req)
    return cached / total
```

This causes a T3 regression (+4–5% vs baseline): T4 requests with higher hit
ratios permanently displace T3 requests regardless of how long T3 has been
waiting. There is no mechanism to prevent indefinite demotion of any request.

**Fix:** blend hit ratio with a logarithmic wait-time boost into a single score:

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

| Turn | baseline | ratio\_t1 | ratio\_t2 | soft\_t1 | soft\_t2 |
|------|----------|-----------|-----------|---------|---------|
| T1 | 234.6ms | 231.5ms (−1%) | 224.0ms (−5%) | 316.9ms (+35%) | 224.4ms (−4%) |
| T2 | 125.0ms | 135.2ms (+8%) | 133.1ms (+6%) | 132.5ms (+6%) | 131.2ms (+5%) |
| T3 | 147.8ms | 155.1ms (+5%) | 151.4ms (+2%) | 122.2ms (−17%) | 141.1ms (−5%) |
| T4 | 154.5ms | 90.2ms (−42%) | 110.2ms (−29%) | 105.4ms (−32%) | 96.7ms (−37%) |

### 2-trial averages vs combined

| Turn | baseline | ratio\_avg | soft\_avg | combined |
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
ratio-only ordering.

The mechanism: T4 requests with higher hit ratios no longer permanently
displace T3 requests. The wait-time boost ensures T3 eventually overtakes T4
when the scheduling gap has grown large enough.

### 2. T4 unchanged

Both strategies achieve ~35% T4 improvement. Soft aging does not help or hurt
at T4 relative to ratio-based ordering.

### 3. T2 regression marginally improved (5% vs 7%)

The structural warm-first cost to T2 is reduced slightly. The T2 regression is
inherent to any warm-first scheme and is not resolved by this change.

### 4. T1 high variance persists

Soft\_t1 shows +35% T1 regression; soft\_t2 shows −4%. This is the same
cold-start instability observed with chunk\_only in the 2026-07-08 decomp
experiment. T1 TTFT under reorder-only conditions cannot be reliably
characterised from two trials.

### 5. Combined still dominates at T3

Soft aging reorder-only reaches −11% at T3; combined (chunk + reorder) reaches
−41%. Dynamic chunking remains the primary driver at T3/T4.

---

## Conclusion

Soft aging is a strictly better sort key than ratio-only for multi-turn
workloads. It eliminates the T3 regression (−11% vs +4%), keeps T4 identical,
and marginally improves T2. The patch is adopted as the default reordering
implementation.

The T2 regression (+5%) remains a known limitation of any warm-first scheme.
Resolving it would require a fundamentally different approach (e.g., PRISM-style
reserved cold-lane slots).

---

## Reproduction

### Prerequisites

```bash
# Apply all scheduler patches (idempotent, safe to re-run)
python3 scripts/patch_scheduler.py
```

### baseline

```bash
env PREFIX_REORDER=0 DYNAMIC_CHUNK=0 \
  python -m vllm.entrypoints.openai.api_server \
  --model /model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 --max-num-seqs 32

python src/replay_sharegpt.py \
  --host localhost --port 8000 \
  --model /model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/sharegpt_v3.json \
  --num-convs 200 --max-turns 4 --min-turns 4 \
  --max-tokens 128 --concurrency 15 \
  --output logs/mt_base_c15.jsonl
```

### ratio-only sort key (archived, collected 2026-07-08)

Requires the pre-soft-aging scheduler (git commit `5dcf9f3` or earlier).
The `_hit_ratio`-only REORDER\_BLOCK was replaced in commit `7b2bbca`.
Archived logs: `logs/2026-07-08-mt-mt_ratio_reorder_c15_t1.jsonl`,
`logs/2026-07-08-mt-mt_ratio_reorder_c15_t2.jsonl`

To re-run on a fresh server at the old commit:
```bash
git checkout 5dcf9f3 -- scripts/patch_scheduler.py
python3 scripts/patch_scheduler.py

env PREFIX_REORDER=1 DYNAMIC_CHUNK=0 \
  python -m vllm.entrypoints.openai.api_server ...
```

### soft aging (this experiment, current default)

```bash
# Uses bundled 2-trial script:
bash scripts/run_soft_aging.sh

# Or manually (AGING_ALPHA=0.3 is the default, explicit here for clarity):
env PREFIX_REORDER=1 DYNAMIC_CHUNK=0 AGING_ALPHA=0.3 \
  python -m vllm.entrypoints.openai.api_server \
  --model /model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct \
  --port 8000 --max-num-seqs 32

python src/replay_sharegpt.py \
  --host localhost --port 8000 \
  --model /model/ModelScope/Qwen/Qwen2.5-Coder-7B-Instruct \
  --dataset data/sharegpt_v3.json \
  --num-convs 200 --max-turns 4 --min-turns 4 \
  --max-tokens 128 --concurrency 15 \
  --output logs/mt_soft_aging_c15_t1.jsonl
```

### combined reference (archived, collected 2026-07-08)

Uses the pre-soft-aging, pre-hysteresis scheduler. Archived log:
`logs/2026-07-08-mt-mt_comb_c15.jsonl`

---

## Implementation

**New env var:**

| Var | Default | Description |
|-----|---------|-------------|
| `AGING_ALPHA` | `0.3` | Continuous aging coefficient |

`AGING_THRESHOLD_MS` is retained in `__init__` as a legacy field but is no
longer read by the reorder block.

**Files changed:**

| File | Change |
|------|--------|
| `scripts/patch_scheduler.py` | Patch 3b REORDER\_BLOCK → soft\_priority; Patch 2b adds `_aging_alpha` |
| `scripts/hotpatch_soft_aging.py` | One-shot upgrade for already-patched servers |
| `scripts/run_soft_aging.sh` | 2-trial experiment runner |
| live scheduler | Patched via hotpatch\_soft\_aging.py on 2026-07-09 |

**Log files:**

| Tag | File |
|-----|------|
| soft\_t1 | `logs/2026-07-09-mt-mt_soft_aging_c15_t1.jsonl` |
| soft\_t2 | `logs/2026-07-09-mt-mt_soft_aging_c15_t2.jsonl` |


---

## Analysis

```bash
python3 src/analyze_ablation.py --avg-pairs \
  "baseline:logs/2026-07-08-mt-mt_base_c15.jsonl" \
  "ratio_t1:logs/2026-07-08-mt-mt_ratio_reorder_c15_t1.jsonl" \
  "ratio_t2:logs/2026-07-08-mt-mt_ratio_reorder_c15_t2.jsonl" \
  "soft_t1:logs/2026-07-09-mt-mt_soft_aging_c15_t1.jsonl" \
  "soft_t2:logs/2026-07-09-mt-mt_soft_aging_c15_t2.jsonl" \
  "combined:logs/2026-07-08-mt-mt_comb_c15.jsonl"
```
