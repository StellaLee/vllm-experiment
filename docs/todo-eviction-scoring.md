# TODO: Eviction Scoring Ablation

## Current implementation

`CF score = (hit_count + 1) / (prefix_depth + 1)`

`prefix_depth` is the block's absolute position from the radix tree root (0 = root).
Shallow blocks score higher and are preserved over deeper blocks.

## Two gaps

**Gap 1 — prefix_depth is a proxy for cascade impact, not a direct measure.**
A block at depth 1 with 50 children has more cascade impact than a block at depth 0
with 1 child, but CF evicts the depth-1 block first.

**Gap 2 — subtree_size doesn't capture utility.**
A block with 50 cold descendants should NOT be preserved over a block with 2 hot
descendants. Large subtree ≠ valuable subtree. Cascade impact only matters if the
subtree is actually being accessed.

## The right metric

The quantity to minimise is **cache utility lost by eviction**. The correct proxy is
`subtree_hit_count` — total cache hits across the block and all its descendants since
last eviction:

```
Score(b) = subtree_hit_count(b) + 1
```

- Cold subtree (subtree_hit_count ≈ 0) → low score → evicted first  ✓
- Hot subtree (many hits) → high score → preserved               ✓
- Large-but-cold subtree → evicted over small-but-hot block      ✓

With recency weighting (if temporal locality matters):
```
Score(b) = (subtree_hit_count(b) + 1) * exp(-lambda * age)
```

## Proposed ablation

Compare three scoring variants on the full 2×2×2 factorial:

| Variant | Formula | Signal |
|---------|---------|--------|
| **Current (CF)** | `(hit_count + 1) / (prefix_depth + 1)` | depth-proxy for descendants |
| **Subtree-size** | `(hit_count + 1) * (subtree_size + 1)` | descendant count, ignores utility |
| **Subtree-hit** | `subtree_hit_count(b) + 1` | actual utility across subtree |

## Implementation cost comparison

| Field | Update trigger | Frequency | Cost per event |
|-------|---------------|-----------|----------------|
| `hit_count` | per-block cache hit | high | O(1) |
| `subtree_size` | block insert / evict | low | O(depth) ancestors |
| `subtree_hit_count` | any cache hit in subtree | high | O(depth) ancestors |

`subtree_hit_count` is more expensive than `subtree_size` because it must update
ancestors on every hit (not just insert/evict). Hits are frequent when the cache is
warm, so O(depth) per hit is the real cost. For depth 5–50, still small relative to
GPU step time, but non-negligible at very high hit rates.

## Implementation note

vLLM's radix tree maintains parent pointers. `subtree_hit_count` maintenance:
- on any block cache hit: walk block → root, increment subtree_hit_count for each ancestor
- on block eviction: walk block → root, subtract this block's hit_count from each ancestor
- reset to 0 on block recycling

Change localized to `kv_cache_utils.py` (`KVCacheBlock` dataclass + `access_block` + `pop_scored_n`).

## Blocks

Main track eviction contribution section. Low priority until CF factorial (ablation B)
confirms CF adds measurable value over LRU at all.
