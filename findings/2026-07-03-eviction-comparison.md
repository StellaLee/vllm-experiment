# Eviction Policy Comparison — 2026-07-03

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-7B-Instruct |
| gpu-memory-utilization | 0.7 |
| Concurrency | 20 |
| Conversations | 50 |
| Dataset | ShareGPT v3 (multi-turn) |
| Max turns | 4 |
| Policies tested | LRU (baseline), TDF λ=0.1, CF |

## Policies

| Policy | Formula | Intuition |
|--------|---------|-----------|
| LRU | least-recently-used (vLLM default) | evict blocks not touched recently |
| TDF | `(hit_count+1)·exp(−λ·age)` | prefer frequently-hit, recently-cached blocks |
| **CF** | `(hit_count+1)/(prefix_depth+1)` | prefer shallow, frequently-hit blocks |

**CF (Cascaded Frequency)**: prefix_depth is the absolute position of a block in its
prefix chain (0 = root block, 1 = second block, …). Evicting a block at depth D makes
all blocks at depths D+1, D+2, … unreachable as prefix hits — a cascade. CF scores
shallow blocks higher so they survive longer regardless of recency.

## Results: Per-Turn TTFT (ms)

| Turn | LRU med | LRU P95 | TDF P95 | CF med | CF P95 | CF vs LRU P95 |
|------|---------|---------|---------|--------|--------|---------------|
| 1 | 158ms | 267ms | 298ms | 66ms | 140ms | +47.4% |
| 2 | 67ms | 197ms | 195ms | 57ms | 75ms | +62.0% |
| 3 | 69ms | 189ms | 209ms | 63ms | 110ms | +41.6% |
| 4 | 102ms | 210ms | 312ms | 64ms | 126ms | +40.2% |

## Key Finding

**CF outperforms LRU on every turn and every metric.**

| Metric | LRU | CF | Improvement |
|--------|-----|----|-------------|
| Turn 1 P95 | 267ms | 140ms | +47.4% |
| Turn 2 P95 | 197ms | 75ms | +62.0% |
| Turn 3 P95 | 189ms | 110ms | +41.6% |
| Turn 4 P95 | 210ms | 126ms | +40.2% |

Turn-4 P95 TTFT improvement: **210ms → 126ms (40.2% faster)**

## Why CF Beats LRU

LRU positions blocks by recency of last access. Under concurrency=20 with
gpu-memory-utilization=0.7, many conversations compete for the KV cache.
When a turn-3 block from conversation A is touched, it jumps to the tail of
the LRU queue — but this can displace a turn-1 block from conversation B that
has not been touched since turn 1 ran. When conversation B reaches turn 4,
its prefix chain is broken and it must recompute from scratch.

CF avoids this by scoring: `(hit_count+1)/(prefix_depth+1)`.

- Turn-1 blocks live at depth 0–3. Even with 0 hits, score ≥ 1/4 = 0.25.
- Turn-3 blocks live at depth 6–9. With 0 hits, score ≤ 1/7 = 0.14.
- CF evicts the deep tail blocks first, keeping the anchor blocks in place.
- When turn-4 arrives, blocks 0–N are all present → full prefix hit → fast TTFT.

### Turn-4 distribution detail

| Policy | Values (ms) |
|--------|-------------|
| LRU | [52, 52, 52, 53, 58, 60, 63, 67, 74, 102, 103, 104, 104, 105, 108, 208, 209, 210, 216] |
| TDF λ=0.1 | [51, 52, 52, 53, 57, 60, 60, 66, 90, 95, 127, 175, 176, 177, 196, 197, 197, 312, 313] |
| CF | [49, 51, 51, 53, 54, 56, 62, 62, 62, 64, 65, 69, 70, 77, 78, 78, 79, 118, 190] |

## Why TDF Failed

TDF penalizes old blocks via `exp(−λ·age)`. Turn-1 blocks are oldest and get
the heaviest decay penalty — the opposite of what the workload needs.
CF has no age term; it rewards shallowness instead, which correctly identifies
the most structurally critical blocks.

## Limitations

- Single run per policy (n=19 turn-4 requests). Results should be replicated
  with more conversations for statistical confidence.
- CF assumes that prefix_depth is a good proxy for "cascade importance."
  This holds for linear prefix chains (ShareGPT) but may not hold for
  tree-structured caches (e.g., speculative decoding, beam search).
- CF's O(n) scan per allocation is acceptable for ~12K blocks but could be
  optimised with a sorted heap for very large KV caches.

## Usage

```bash
# Apply patches once to vLLM 0.23.0:
bash patches/apply_tdf.sh

# Run with CF:
EVICTION_POLICY=cf python -m vllm.entrypoints.api_server ...

# Run with TDF:
EVICTION_POLICY=tdf TDF_LAMBDA=0.1 python -m vllm.entrypoints.api_server ...

# Run with LRU (default):
python -m vllm.entrypoints.api_server ...
```


## Reproduction

**Prerequisites:**
```bash
bash scripts/setup.sh          # fetches ShareGPT V3 dataset
bash patches/apply_patches.sh  # patches vLLM 0.23.0 in-place
```

**Run (50-conversation pilot — same size as this run):**
```bash
NUM_CONVS=50 bash scripts/run_full_comparison.sh
```

Output written to `findings/YYYY-MM-DD-full-eviction-comparison.md`.

Note: this was a 50-conversation pilot (n=19 turn-4 samples). For statistically
robust results see [`2026-07-03-full-eviction-comparison.md`](2026-07-03-full-eviction-comparison.md),
which used 200 conversations.
