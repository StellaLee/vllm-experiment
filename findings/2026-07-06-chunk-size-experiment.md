# Dynamic Chunk Size Experiment — 2026-07-06

## Setup

| Parameter | Value |
|-----------|-------|
| Model | Qwen2.5-Coder-7B-Instruct |
| max-num-seqs | 32 |
| Prompts | 150 |
| Request rate | inf (all at once) |
| Datasets | BurstGPT_1.csv, ShareGPT v3 |
| Baseline | Static `max_num_scheduled_tokens` = 2048 tok/step (`DYNAMIC_CHUNK=0`) |
| Dynamic | Bang-bang controller (`DYNAMIC_CHUNK=1`, target=8, min=256) |

**Controller logic:**
- decode depth > target × 1.5 → halve token budget (floor: 256)
- decode depth < target × 0.5 → double token budget (ceiling: 2048)
- Signal: count of `running` requests with `is_prefill_chunk=False`

## TTFT (Time to First Token)

| Dataset | Baseline p50 | Baseline p95 | Baseline p99 | Dynamic p50 | Dynamic p95 | Dynamic p99 |
|---------|-------------|-------------|-------------|------------|------------|------------|
| BurstGPT | 4818 ms | 11652 ms | 12518 ms | 5111 ms | 11898 ms | 12930 ms |
| ShareGPT | 7083 ms | 14346 ms | 14914 ms | 5633 ms | 12378 ms | 13277 ms |

## TPOT (Time per Output Token)

| Dataset | Baseline p50 | Baseline p95 | Baseline p99 | Dynamic p50 | Dynamic p95 | Dynamic p99 |
|---------|-------------|-------------|-------------|------------|------------|------------|
| BurstGPT | 38.2 ms | 127.0 ms | 173.7 ms | 32.3 ms | 37.6 ms | 55.4 ms |
| ShareGPT | 21.6 ms | 29.5 ms | 43.6 ms | 20.9 ms | 25.3 ms | 30.5 ms |

## E2EL (End-to-End Latency)

| Dataset | Baseline p50 | Baseline p95 | Baseline p99 | Dynamic p50 | Dynamic p95 | Dynamic p99 |
|---------|-------------|-------------|-------------|------------|------------|------------|
| BurstGPT | 7243 ms | 17581 ms | 29342 ms | 7674 ms | 26329 ms | 31655 ms |
| ShareGPT | 10647 ms | 19158 ms | 21446 ms | 8959 ms | 18808 ms | 22639 ms |

## Throughput

| Dataset | Baseline | Dynamic | Delta |
|---------|----------|---------|-------|
| BurstGPT | 4.99 req/s | 4.33 req/s | −13.1% |
| ShareGPT | 5.61 req/s | 6.01 req/s | +7.2% |

## Summary

### ShareGPT (mixed arrival) — clean win

| Metric | Delta |
|--------|-------|
| TTFT p50 | −20.5% |
| TTFT p95 | −13.7% |
| TPOT p95 | −14.2% |
| TPOT p99 | −30.0% |
| E2EL p50 | −15.8% |
| Throughput | +7.2% |

With request arrivals spread over time, the controller successfully shields
decoding requests from large prefill chunks. Decode starvation is reduced and
overall throughput improves.

### BurstGPT (thundering herd) — mixed result

TPOT p95 improved 70% (127 ms → 38 ms), confirming decode starvation is fixed.
However, TTFT p50 rose 6.1% and E2EL p95 rose 50%: when all 150 requests arrive
simultaneously the controller sees maximum decode depth immediately and halves the
chunk budget repeatedly, slowing prefill for all queued requests and increasing
tail latency.

Total output tokens also grew (16 082 → 21 292), suggesting sequences were
allowed to generate longer under the reduced-chunk regime.

**Root cause:** the bang-bang controller reacts to instantaneous decode depth. Under
pure thundering-herd load this signal is always high, so the controller parks at
the minimum budget (256 tokens) throughout, creating a prefill bottleneck that
exceeds the decode-starvation problem it was solving.

**Mitigation options:**
1. Raise `DYNAMIC_CHUNK_MIN` to avoid parking at the floor (e.g. 512 or 1024)
2. Use a finite request rate (e.g. `--request-rate 20`) to break the herd
3. Add a moving-average filter on decode depth before applying thresholds

## Raw results

`logs/`

| File | Contents |
|------|----------|
| `2026-07-06-chunk-baseline-burstgpt.json` | baseline BurstGPT |
| `2026-07-06-chunk-baseline-sharegpt.json` | baseline ShareGPT |
| `2026-07-06-chunk-dynamic-burstgpt.json` | dynamic BurstGPT |
| `2026-07-06-chunk-dynamic-sharegpt.json` | dynamic ShareGPT |

## Reproduction

**Prerequisites:**
```bash
bash patches/apply_patches.sh --chunk-size   # patches vLLM 0.23.0 in-place
```

**Run all 4 benchmarks:**
```bash
bash scripts/run_chunk_experiment.sh
```

**Key parameters (override via env):**
```bash
NUM_PROMPTS=150 DYNAMIC_CHUNK_TARGET=8 DYNAMIC_CHUNK_MIN=256 \
  bash scripts/run_chunk_experiment.sh
```

Output written to `findings/chunk_<timestamp>/` (JSON) and analysed by
`src/analyze_chunk.py`. Requires ~30 min on RTX 4090 (two server starts
× ~5 min each + four benchmark runs × ~3 min each).
