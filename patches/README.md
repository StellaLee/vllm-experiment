# vLLM 0.23.0 Patches

Two independent sets of patches are provided. Each can be applied standalone.

---

## 1. Pluggable KV-Cache Eviction Policy

Adds CF (Cascaded Frequency) and TDF (Time-Decayed Frequency) eviction policies
to vLLM's prefix-cache block pool, controllable via `EVICTION_POLICY`.

**Files patched:** `vllm/v1/core/kv_cache_utils.py`, `vllm/v1/core/block_pool.py`

| Policy | `EVICTION_POLICY` | Score formula | Best for |
|--------|-------------------|---------------|----------|
| LRU | `lru` (default) | least-recently-used | general workloads |
| Cascaded Frequency | `cf` | `(hit_count+1)/(prefix_depth+1)` | multi-turn chat |
| Time-Decayed Frequency | `tdf` | `(hit_count+1)·exp(−λ·age)` | repeated single-turn |

**Result:** CF outperformed LRU by 40–62% on P95 TTFT in ShareGPT multi-turn
experiments (concurrency=20, Qwen2.5-Coder-7B, gpu-mem-util=0.7).
See `findings/2026-07-03-eviction-comparison.md`.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVICTION_POLICY` | `lru` | `lru`, `cf`, or `tdf` |
| `TDF_LAMBDA` | `0.1` | Decay rate λ for TDF |

---

## 2. Dynamic Chunk Size Controller

Adds a bang-bang feedback controller to vLLM's chunked-prefill scheduler.
Instead of a fixed token budget per scheduling step, the budget shrinks when
many decode-phase requests are queued and grows when the queue is shallow.
This prevents **decode starvation** under mixed prefill/decode workloads.

**File patched:** `vllm/v1/core/sched/scheduler.py`

**Mechanism:**
- Signal: count of `running` requests with `is_prefill_chunk=False` (decode depth)
- High threshold `> target × 1.5` → halve chunk (floor: `DYNAMIC_CHUNK_MIN`)
- Low threshold `< target × 0.5` → double chunk (ceiling: static `max_num_scheduled_tokens`)

**Result (Qwen2.5-Coder-7B, 150 prompts, rate=inf):**

| Dataset | TTFT p50 | TTFT p95 | TPOT p95 | Throughput |
|---------|----------|----------|----------|------------|
| ShareGPT baseline | 7083 ms | 14346 ms | 29.5 ms | 5.61 req/s |
| ShareGPT dynamic  | 5633 ms | 12378 ms | 25.3 ms | 6.01 req/s |
| Delta | **−20.5%** | **−13.7%** | **−14.2%** | **+7.2%** |

BurstGPT (thundering-herd arrival) reduced TPOT p95 by 70% but increased
E2EL tail due to the controller over-shrinking the chunk under pure burst load.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DYNAMIC_CHUNK` | `0` | Set to `1` to enable |
| `DYNAMIC_CHUNK_TARGET` | `8` | Target decode-queue depth |
| `DYNAMIC_CHUNK_MIN` | `256` | Minimum token budget per step |

---

## Applying patches

```bash
# Apply all patches at once
bash patches/apply_patches.sh

# Or apply individually
bash patches/apply_patches.sh --eviction-only
bash patches/apply_patches.sh --chunk-only
```

## Reverting

```bash
VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
patch -R -p1 -d "$VLLM_SITE/.." < patches/kv_cache_utils.patch
patch -R -p1 -d "$VLLM_SITE/.." < patches/block_pool.patch
patch -R -p1 -d "$VLLM_SITE/.." < patches/scheduler.patch
```
