import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                                 "scripts", "eenergy", "router"))
from cache_mirror import block_hashes, new_tokens_if_routed, record_cached, BLOCK_SIZE  # noqa: E402


def test_block_hashes_ignores_partial_trailing_block():
    ids = list(range(BLOCK_SIZE + 3))  # one full block + 3 leftover tokens
    hashes = block_hashes(ids)
    assert len(hashes) == 1


def test_new_tokens_with_empty_cache_is_full_prompt_length():
    ids = list(range(BLOCK_SIZE * 2 + 1))
    assert new_tokens_if_routed(ids, cached_hashes=set()) == len(ids)


def test_record_cached_then_identical_prefix_reduces_new_tokens():
    cache = set()
    first = list(range(BLOCK_SIZE * 2))       # exactly 2 full blocks
    record_cached(first, cache)
    second = first + list(range(BLOCK_SIZE * 2, BLOCK_SIZE * 2 + 5))  # same prefix + 5 more
    # 2 full blocks already cached (2*BLOCK_SIZE tokens); only the trailing 5 are new
    assert new_tokens_if_routed(second, cache) == 5


def test_diverging_prefix_after_first_block_only_credits_matching_prefix():
    cache = set()
    first = list(range(BLOCK_SIZE * 2))
    record_cached(first, cache)
    # second request shares block 0 exactly, diverges inside block 1
    diverged = first[:BLOCK_SIZE] + [-1] * BLOCK_SIZE
    result = new_tokens_if_routed(diverged, cache)
    assert result == BLOCK_SIZE  # only block 1 (the diverging one) is new
