# TODO: Eviction Scoring Ablation

## Current implementation

`CF score = (hit_count + 1) / (prefix_depth + 1)`

`prefix_depth` is the block's absolute position from the radix tree root (0 = root).
Shallow blocks score higher and are preserved over deeper blocks.

## The gap

`prefix_depth` is a proxy for cascade impact, not a direct measure.
It assumes shallow ↔ many descendants, which holds for typical workloads but not always.

A block at depth 1 with 50 children has more cascade impact than a block at depth 0
with 1 child — but CF evicts the depth-1 block first.

## Proposed ablation

Compare two scoring variants on the full 2×2×2 factorial:

| Variant | Formula | Signal |
|---------|---------|--------|
| **Current (CF)** | `(hit_count + 1) / (prefix_depth + 1)` | depth from root as proxy for descendants |
| **Subtree-weighted** | `(hit_count + 1) * (subtree_size + 1)` | actual descendant count in cache |

`subtree_size` = number of descendant blocks currently live in the radix cache that
depend on this block. Requires tracking on allocation (+1 for each ancestor) and
eviction (−1 for each ancestor).

## Implementation note

vLLM's radix tree maintains parent pointers. Adding `subtree_size` means:
- increment all ancestors by 1 on block insertion
- decrement all ancestors by 1 on block eviction
- cost: O(depth) per event — negligible

Change is localized to `kv_cache_utils.py` (`KVCacheBlock` dataclass + `pop_scored_n`).

## Expected outcome

Subtree-weighted should outperform depth-proxy under irregular prefix trees
(e.g., coding agent workloads with branching context). On ShareGPT / BurstGPT
(mostly linear chains) the difference may be small.

## Blocks

Main track eviction contribution section. Low priority until CF factorial (ablation B)
confirms CF adds measurable value over LRU.
