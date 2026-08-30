"""Per-replica prefix-cache simulator: lets the router compute how many NEW tokens a
request would need if routed to a given replica, without querying that replica -- mirrors
how LMETRIC and prefix-aware routers generally avoid a round-trip per routing decision.

Uses a rolling block-hash chain (hash of block k depends on the hash of block k-1 and
block k's own tokens), the same structural idea vLLM's own prefix-caching uses: a hash
match at block k implies the ENTIRE prefix up to and including block k matches, so the
router only needs to walk forward from block 0 until the first miss."""

BLOCK_SIZE = 16  # tokens per cache block


def block_hashes(token_ids: list, block_size: int = BLOCK_SIZE) -> list:
    """Rolling prefix-block hashes. A partial trailing block (fewer than block_size tokens)
    is never included -- it can't be a cache hit target for a later, longer request until
    it's actually completed to a full block by more tokens."""
    hashes = []
    prev = 0
    for start in range(0, len(token_ids), block_size):
        block = tuple(token_ids[start:start + block_size])
        if len(block) < block_size:
            break
        prev = hash((prev, block))
        hashes.append(prev)
    return hashes


def new_tokens_if_routed(token_ids: list, cached_hashes: set,
                          block_size: int = BLOCK_SIZE) -> int:
    """How many of token_ids are NOT covered by cached_hashes if routed to the replica that
    owns cached_hashes. Walks the hash chain from the start; stops at the first block whose
    hash isn't in cached_hashes (by construction, everything before that block matched)."""
    hashes = block_hashes(token_ids, block_size)
    matched_blocks = 0
    for h in hashes:
        if h in cached_hashes:
            matched_blocks += 1
        else:
            break
    return len(token_ids) - matched_blocks * block_size


def record_cached(token_ids: list, cached_hashes: set,
                   block_size: int = BLOCK_SIZE) -> None:
    """Mark this request's full blocks as cached on the replica. Call once the router has
    dispatched the request there."""
    cached_hashes.update(block_hashes(token_ids, block_size))
