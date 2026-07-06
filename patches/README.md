# vLLM 0.23.0 Patches

Two independent patch sets, one directory per feature.

```
patches/
  eviction/       -- pluggable KV-cache eviction policy
  chunk_size/     -- dynamic chunked-prefill token budget
  apply_patches.sh
  README.md
```

## Applying

```bash
bash patches/apply_patches.sh                # both
bash patches/apply_patches.sh --eviction     # eviction only
bash patches/apply_patches.sh --chunk-size   # chunk size only
```

## Reverting

```bash
VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
# eviction
patch -R -p1 -d "$VLLM_SITE/.." < patches/eviction/kv_cache_utils.patch
patch -R -p1 -d "$VLLM_SITE/.." < patches/eviction/block_pool.patch
# chunk size
patch -R -p1 -d "$VLLM_SITE/.." < patches/chunk_size/scheduler.patch
```

---

## eviction/ — Pluggable KV-Cache Eviction Policy

**Files patched:** `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/block_pool.py`

Adds CF (Cascaded Frequency) and TDF (Time-Decayed Frequency) eviction policies
alongside vLLM's default LRU, selectable via `EVICTION_POLICY`.

| Policy | `EVICTION_POLICY` | Score | Best for |
|--------|-------------------|-------|----------|
| LRU | `lru` (default) | least-recently-used | general |
| Cascaded Frequency | `cf` | `(hit_count+1)/(prefix_depth+1)` | multi-turn chat |
| Time-Decayed Frequency | `tdf` | `(hit_count+1)·exp(−λ·age)` | repeated single-turn |

**Result:** CF outperformed LRU by 40–62% on P95 TTFT in ShareGPT multi-turn
experiments (concurrency=20, Qwen2.5-Coder-7B).
See `findings/2026-07-03-eviction-comparison.md`.

| Variable | Default | Description |
|----------|---------|-------------|
| `EVICTION_POLICY` | `lru` | `lru`, `cf`, or `tdf` |
| `TDF_LAMBDA` | `0.1` | Decay rate λ for TDF |

---

## chunk_size/ — Dynamic Chunked-Prefill Token Budget

**File patched:** `vllm/v1/core/sched/scheduler.py`

Adds `ChunkSizeController`, a bang-bang feedback loop that adjusts the
per-step token budget based on decode-queue depth, preventing decode
starvation under mixed prefill/decode workloads.

- decode depth > `TARGET × 1.5` → halve budget (floor: `DYNAMIC_CHUNK_MIN`)
- decode depth < `TARGET × 0.5` → double budget (ceiling: static max)

**Result (Qwen2.5-Coder-7B, 150 prompts, rate=inf):**

| Dataset | TTFT p50 | TTFT p95 | TPOT p95 | Throughput |
|---------|----------|----------|----------|------------|
| ShareGPT baseline | 7083 ms | 14346 ms | 29.5 ms | 5.61 req/s |
| ShareGPT dynamic  | 5633 ms | 12378 ms | 25.3 ms | 6.01 req/s |
| Delta | **−20.5%** | **−13.7%** | **−14.2%** | **+7.2%** |

See `findings/chunk_20260706_093754/`.

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMIC_CHUNK` | `0` | Set to `1` to enable |
| `DYNAMIC_CHUNK_TARGET` | `8` | Target decode-queue depth |
| `DYNAMIC_CHUNK_MIN` | `256` | Minimum token budget per step |
