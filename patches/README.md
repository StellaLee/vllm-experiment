# KV-Cache Eviction Policy Patches for vLLM 0.23.0

These patches add a pluggable eviction policy to vLLM's prefix-cache block
pool. Three policies are available via the `EVICTION_POLICY` environment
variable.

## Policies

| Policy | `EVICTION_POLICY` | Score formula | Best for |
|--------|-------------------|---------------|----------|
| LRU | `lru` (default) | least-recently-used | general workloads |
| Cascaded Frequency | `cf` | `(hit_count+1)/(prefix_depth+1)` | multi-turn chat |
| Time-Decayed Frequency | `tdf` | `(hit_count+1)·exp(−λ·age)` | single-turn repeated queries |

**CF** outperformed LRU by 40–62% on P95 TTFT across all turns in ShareGPT
multi-turn experiments (concurrency=20, Qwen2.5-Coder-7B, gpu-mem-util=0.7).
See `findings/2026-07-03-eviction-comparison.md` for full results.

## Prerequisites

- Python 3.10+
- vLLM 0.23.0 installed (`pip install vllm==0.23.0`)
- `patch` command available (`apt install patch` or `brew install patch`)

## Quick start

```bash
git clone https://github.com/StellaLee/vllm-experiment.git
cd vllm-experiment

# 1. Install vLLM 0.23.0 if not already present
pip install vllm==0.23.0

# 2. Apply the patches (auto-detects vLLM install location)
bash patches/apply_patches.sh

# 3. Start the server with CF eviction
EVICTION_POLICY=cf python -m vllm.entrypoints.api_server \
  --model <your-model> \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.7 \
  --port 8000
```

## What the patches change

Two files inside the vLLM package are modified:

### `vllm/v1/core/kv_cache_utils.py`
- **`KVCacheBlock`** gains three new fields:
  - `hit_count: int` — incremented each time this block is reused as a prefix cache hit
  - `last_alloc_time: float` — monotonic timestamp when the block entered the prefix cache (used by TDF)
  - `prefix_depth: int` — absolute position of the block in its prefix chain (0 = root block)
- **`FreeKVCacheBlockQueue.pop_scored_n(n, policy, ...)`** — new method that scans all free blocks, scores each by the selected policy, and removes the `n` lowest-scored ones

### `vllm/v1/core/block_pool.py`
- Reads `EVICTION_POLICY` and `TDF_LAMBDA` env vars at startup
- `get_new_blocks()` routes to `pop_scored_n()` when policy is `cf` or `tdf`
- `cache_full_blocks()` stamps `last_alloc_time` and `prefix_depth` when a block enters the cache
- `touch()` increments `hit_count` on prefix cache hits
- `_maybe_evict_cached_block()` resets all three fields on eviction

## Reverting

To undo the patches (restore original vLLM behavior):

```bash
VLLM_SITE=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))")
patch -R -p1 -d "$VLLM_SITE/.." < patches/kv_cache_utils.patch
patch -R -p1 -d "$VLLM_SITE/.." < patches/block_pool.patch
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EVICTION_POLICY` | `lru` | Eviction policy: `lru`, `cf`, or `tdf` |
| `TDF_LAMBDA` | `0.1` | Decay rate λ for TDF policy |
