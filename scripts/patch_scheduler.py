#!/usr/bin/env python3
"""Apply PREFIX_REORDER / DYNAMIC_CHUNK / AGING_THRESHOLD_MS patches to the
vLLM v1 scheduler. Safe to re-run — skips hunks that are already present."""

import sys
from pathlib import Path

SCHED = Path("/root/miniconda3/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py")
if not SCHED.exists():
    print(f"ERROR: {SCHED} not found", file=sys.stderr)
    sys.exit(1)

src = SCHED.read_text()
changed = False

# ── Patch 1: ChunkSizeController class (insert before class Scheduler) ──────
CHUNK_CLASS = '''
class ChunkSizeController:
    """Bang-bang controller for dynamic chunked-prefill token budget.

    Shrinks chunk size when decode queue is deep, grows it when shallow.
    Prevents decode starvation under mixed prefill/decode workloads.

    Enable: DYNAMIC_CHUNK=1
    Tune:   DYNAMIC_CHUNK_TARGET (int, default 8)
            DYNAMIC_CHUNK_MIN    (int, default 256)
    """

    def __init__(self, min_tokens: int, max_tokens: int, target: int) -> None:
        self.chunk = max_tokens
        self.min = min_tokens
        self.max = max_tokens
        self.target = target
        self._step_count = 0

    def step(self, decode_depth: int) -> int:
        self._step_count += 1
        if decode_depth > self.target * 1.5:
            self.chunk = max(self.min, self.chunk // 2)
        elif decode_depth < self.target * 0.5:
            self.chunk = min(self.max, self.chunk * 2)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).debug(
                "ChunkSizeController step=%d decode_depth=%d chunk=%d",
                self._step_count, decode_depth, self.chunk,
            )
        return self.chunk

'''

if "class ChunkSizeController" not in src:
    src = src.replace(
        "class Scheduler(SchedulerInterface):",
        CHUNK_CLASS + "class Scheduler(SchedulerInterface):",
    )
    changed = True
    print("Patch 1 applied: ChunkSizeController class")
else:
    print("Patch 1 already present: ChunkSizeController class")

# ── Patch 2: __init__ wiring ─────────────────────────────────────────────────
INIT_PATCH = """
        # Dynamic chunk size controller (DYNAMIC_CHUNK=1 to enable).
        _dynamic_chunk = os.getenv("DYNAMIC_CHUNK", "0").strip() == "1"
        if _dynamic_chunk:
            _target = int(os.getenv("DYNAMIC_CHUNK_TARGET", "8"))
            _min = int(os.getenv("DYNAMIC_CHUNK_MIN", "256"))
            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(
                min_tokens=_min,
                max_tokens=self.max_num_scheduled_tokens,
                target=_target,
            )
            logger.info(
                "Dynamic chunk size enabled: min=%d max=%d target=%d",
                _min, self.max_num_scheduled_tokens, _target,
            )
        else:
            self._chunk_ctrl = None
        _prefix_reorder = os.getenv("PREFIX_REORDER", "0").strip() == "1"
        self._prefix_reorder = _prefix_reorder
        self._aging_threshold_ms = float(os.getenv("AGING_THRESHOLD_MS", "inf"))
        if _prefix_reorder:
            logger.info(
                "Prefix-aware request reordering enabled (aging %.0f ms)",
                self._aging_threshold_ms,
            )
"""

if "_dynamic_chunk = os.getenv" not in src:
    anchor = "            else self.scheduler_config.max_num_batched_tokens\n        )"
    if anchor not in src:
        print("ERROR: Patch 2 anchor not found — check scheduler version", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor, anchor + "\n" + INIT_PATCH)
    changed = True
    print("Patch 2 applied: __init__ wiring")
else:
    print("Patch 2 already present: __init__ wiring")

# ── Patch 3a: chunk controller step in schedule() ────────────────────────────
CHUNK_STEP = """
        if self._chunk_ctrl is not None:
            _decode_depth = sum(
                1 for r in self.running if not r.is_prefill_chunk
            )
            token_budget = self._chunk_ctrl.step(_decode_depth)
"""

if "self._chunk_ctrl is not None" not in src:
    anchor_chunk = "        token_budget = self.max_num_scheduled_tokens\n"
    if anchor_chunk not in src:
        print("ERROR: Patch 3a anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor_chunk, anchor_chunk + CHUNK_STEP)
    changed = True
    print("Patch 3a applied: chunk controller step in schedule()")
else:
    print("Patch 3a already present: chunk controller step")

# ── Patch 3b: warm-first reorder block in schedule() ─────────────────────────
REORDER_BLOCK = """            if self._prefix_reorder and self.waiting and hasattr(self.waiting, 'extendleft'):
                import time as _time
                _now = _time.time()
                _thresh_s = self._aging_threshold_ms / 1000.0
                def _cached_tokens(req):
                    if req.num_computed_tokens > 0:
                        return req.num_computed_tokens
                    _, n = self.kv_cache_manager.get_computed_blocks(req)
                    return n
                _aged = sorted(
                    [r for r in self.waiting if (_now - r.arrival_time) >= _thresh_s],
                    key=lambda r: r.arrival_time,
                )
                _fresh = sorted(
                    [r for r in self.waiting if (_now - r.arrival_time) < _thresh_s],
                    key=_cached_tokens, reverse=True,
                )
                self.waiting.clear()
                self.waiting.extend(_aged + _fresh)
"""

if "self._prefix_reorder and self.waiting" not in src:
    anchor_reorder = "            step_skipped_waiting = create_request_queue(self.policy)\n"
    if anchor_reorder not in src:
        print("ERROR: Patch 3b anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor_reorder, anchor_reorder + REORDER_BLOCK)
    changed = True
    print("Patch 3b applied: warm-first reorder block in schedule()")
else:
    print("Patch 3b already present: warm-first reorder block")

if changed:
    SCHED.write_text(src)
    print(f"\nWrote patched scheduler to {SCHED}")
else:
    print("\nAll patches already present — no changes written.")
