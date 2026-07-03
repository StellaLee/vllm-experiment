# Eviction Policy Comparison — 2026-07-03

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-7B-Instruct |
| gpu-memory-utilization | 0.7 |
| Concurrency | 20 |
| Conversations | 50 (num_convs) |
| Dataset | ShareGPT v3 (multi-turn) |
| Max turns | 4 |
| Policies tested | LRU (baseline), TDF λ=0.1, TDF λ=0.001 |
| TDF score formula | (hit_count+1) × exp(−λ × age_seconds) |

## Per-Turn TTFT — All Policies (ms)

| Turn | LRU med | LRU P95 | TDF λ=0.1 med | TDF λ=0.1 P95 | TDF λ=0.001 med | TDF λ=0.001 P95 |
|------|---------|---------|--------------|--------------|----------------|----------------|
| 1 | 158ms | 267ms | 115ms | 298ms | 104ms | 312ms |
| 2 | 67ms | 197ms | 90ms | 195ms | 70ms | 212ms |
| 3 | 69ms | 189ms | 79ms | 209ms | 69ms | 351ms |
| 4 | 102ms | 210ms | 95ms | 312ms | 130ms | 404ms |

## Key Finding

Both TDF variants underperform LRU at turn 4 (the most eviction-sensitive turn):

| Policy | Turn-4 P95 TTFT | vs LRU |
|--------|----------------|--------|
| LRU | 210ms | — |
| TDF λ=0.1 | 312ms | +102ms (+48.6%) |
| TDF λ=0.001 | 404ms | +194ms (+92.5%) |

## Root Cause Analysis

### Why TDF underperforms LRU on ShareGPT multi-turn

The ShareGPT workload has a distinctive structure: each turn N sends the full
accumulated context (turns 1..N-1) as a prefix. This means turn-1 blocks are the
**oldest** blocks in the KV cache, but also the **most critical** — every subsequent
turn of the same conversation needs them as a prefix.

**LRU implicitly solves this**: when turn-2 hits turn-1 blocks as prefix, `touch()`
is called, incrementing ref_cnt and removing those blocks from the free queue. When
the request completes, the blocks are freed back to the **tail** of the queue (most
recently used → last evicted). This recency refreshing happens on every prefix hit,
naturally protecting active conversation blocks.

**TDF fails because `last_alloc_time` represents when the block entered the prefix
cache, not when it was last accessed**. Under TDF:

- Score(b) = (hit_count+1) × exp(−λ × age)
- With λ=0.1 and age=30s: exp(−3) ≈ 0.05
  Even a 10-hit block scores 0.55, while a brand-new 0-hit block scores 1.0
- TDF preferentially evicts old blocks even when they are the most-needed prefix blocks

With λ=0.001 (near-LFU), the age penalty is negligible, but hit_count-only selection
still performs worse than LRU. Likely cause: at the moment eviction is needed,
turn-1 prefix blocks of long conversations may have accumulated fewer hits than
shorter conversations whose blocks have been freed and reallocated multiple times.

### Key structural mismatch

TDF is designed for workloads where **frequency of past access predicts future
value**. In ShareGPT multi-turn:
- Frequency of past access ✓ (multi-hit prefix blocks ARE valuable)
- But **age is inversely correlated with value**: oldest blocks = most prefix turns

LRU's recency signal happens to be a near-perfect proxy for "is this block currently
part of an active conversation?" — because every prefix hit refreshes the LRU order.

## What Would Beat LRU Here

A policy that explicitly tracks **which conversation a block belongs to** and scores
based on conversation-level recency (e.g., "this block belongs to a conversation that
last sent a request 2 seconds ago" = high value) would likely outperform LRU. This
requires per-request metadata at the block level, which vLLM does not currently expose.

Alternative: **LRU-K** (evict based on K-th most recent access rather than most
recent) could be more robust to one-off touches. But for this specific workload,
standard LRU appears near-optimal.

## Distribution Details (Turn 4)

| Policy | Min | Median | P95 | Max | Fast (<100ms) | Slow (>200ms) |
|--------|-----|--------|-----|-----|---------------|---------------|
| LRU | 52ms | 102ms | 210ms | 216ms | 9/19 | 4/19 |
| TDF λ=0.1 | 51ms | 95ms | 312ms | 313ms | 10/19 | 2/19 |
| TDF λ=0.001 | 52ms | 130ms | 404ms | 405ms | 8/19 | 8/19 |

## Conclusion

For ShareGPT-style multi-turn conversations at concurrency=20, **LRU is superior to
TDF** at all λ values tested. The time-decay term in TDF directly penalizes the blocks
that are most critical for multi-turn prefix caching.

The TDF policy is a better fit for workloads where:
1. Requests are mostly single-turn (no accumulative prefix)
2. Popular prefixes recur across users, not just within one conversation
3. Cache size is large enough that only true "cold" content needs eviction

Next steps:
1. Test TDF on a document-retrieval / RAG workload where the same documents are
   repeatedly queried by different users — TDF's frequency signal would shine there
2. Explore "conversation-aware LRU" that tracks request timestamps per conversation
   and evicts blocks from conversations that have been idle the longest

