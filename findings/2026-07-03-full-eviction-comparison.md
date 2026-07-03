# Eviction Policy Comparison — 2026-07-03

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-7B-Instruct |
| gpu-memory-utilization | 0.7 |
| Concurrency | 20 |
| Conversations | 200 (all ≥4 turns) |
| Dataset | ShareGPT v3 |
| Max turns | 4 |
| Policies | LRU (baseline), TDF λ=0.1, CF |

| Policy | Score formula |
|--------|---------------|
| LRU | least-recently-used (vLLM default) |
| TDF | `(hit_count+1)·exp(−λ·age)` |
| **CF** | `(hit_count+1)/(prefix_depth+1)` |

## TTFT (Time to First Token)

| Turn | n | LRU med | LRU P95 | TDF P95 | CF med | CF P95 | CF vs LRU P95 |
|------|---|---------|---------|---------|--------|--------|---------------|
| 1 | 200 | 57.5ms | 166.7ms | 124.6ms | 56.9ms | 127.9ms | +23.3% |
| 2 | 200 | 58.4ms | 149.8ms | 165.0ms | 58.1ms | 122.7ms | +18.1% |
| 3 | 200 | 58.0ms | 102.6ms | 99.9ms | 58.1ms | 97.5ms | +5.0% |
| 4 | 200 | 58.0ms | 108.1ms | 110.6ms | 58.0ms | 99.8ms | +7.7% |

## TPOT (Time per Output Token, decode phase only)

TPOT = (latency − TTFT) / output_words. Output words used as token proxy (~0.75 words/token).
Lower TPOT = higher decode throughput.

| Turn | LRU med | LRU P95 | TDF P95 | CF med | CF P95 | CF vs LRU P95 |
|------|---------|---------|---------|--------|--------|---------------|
| 1 | 23.0ms | 55.8ms | 57.2ms | 23.0ms | 55.3ms | +0.9% |
| 2 | 23.1ms | 39.5ms | 39.4ms | 23.1ms | 39.5ms | +0.1% |
| 3 | 23.3ms | 37.5ms | 37.5ms | 23.2ms | 37.8ms | -0.8% |
| 4 | 23.0ms | 37.3ms | 37.7ms | 22.9ms | 37.8ms | -1.4% |

## Turn-4 Summary

| Metric | LRU | CF | Improvement |
|--------|-----|----|-------------|
| P95 TTFT | 108.1ms | 99.8ms | +7.7% |
| Median TTFT | 58.0ms | 58.0ms | +0.1% |
| P95 TPOT | 37.3ms | 37.8ms | -1.4% |
| Median TPOT | 23.0ms | 22.9ms | +0.4% |

## Raw counts

| Turn | LRU | CF |
|------|-----|-----|
| 1 | 200 | 200 |
| 2 | 200 | 200 |
| 3 | 200 | 200 |
| 4 | 200 | 200 |
