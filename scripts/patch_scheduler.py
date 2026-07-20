#!/usr/bin/env python3
"""Apply PREFIX_REORDER / DYNAMIC_CHUNK / AGING_ALPHA patches to the
vLLM v1 scheduler. Safe to re-run — skips hunks that are already present.

Env vars:
  PREFIX_REORDER=1         enable warm-first soft-aging reorder
  DYNAMIC_CHUNK=1          enable bang-bang chunk size controller with hysteresis
  AGING_ALPHA=0.3          soft-aging coefficient (default 0.3)
                           score = hit_ratio + AGING_ALPHA * log(1 + wait_s)
  AGING_THRESHOLD_MS=inf   (legacy, no longer used by reorder block)
  DYNAMIC_CHUNK_TARGET=8   decode queue depth target
  DYNAMIC_CHUNK_MIN=256    minimum chunk size (tokens)
  DYNAMIC_CHUNK_HOLD=3     consecutive steps above/below threshold before resize
"""

import os
import sys
from pathlib import Path


def _find_sched() -> Path:
    """Locate scheduler.py in the *active* vLLM install (env override wins)."""
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415 -- resolved against the running interpreter
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


SCHED = _find_sched()
if not SCHED.exists():
    print(f"ERROR: {SCHED} not found", file=sys.stderr)
    sys.exit(1)

src = SCHED.read_text()
changed = False

# ── Patch 1: ChunkSizeController class (insert before class Scheduler) ──────
CHUNK_CLASS = '''
class ChunkSizeController:
    """Bang-bang controller with hysteresis for dynamic chunked-prefill.

    Shrinks chunk when decode queue stays deep for HOLD consecutive steps.
    Grows when decode queue stays shallow for HOLD consecutive steps.
    Holding prevents rapid chunk oscillation at the boundary.

    Enable: DYNAMIC_CHUNK=1
    Tune:   DYNAMIC_CHUNK_TARGET (int, default 8)
            DYNAMIC_CHUNK_MIN    (int, default 256)
            DYNAMIC_CHUNK_HOLD   (int, default 3)
    """

    def __init__(self, min_tokens: int, max_tokens: int, target: int,
                 hold: int = 3) -> None:
        self.chunk = max_tokens
        self.min = min_tokens
        self.max = max_tokens
        self.target = target
        self.hold = hold
        self._step_count = 0
        self._shrink_count = 0
        self._grow_count = 0

    def step(self, decode_depth: int) -> int:
        self._step_count += 1
        if decode_depth > self.target * 1.5:
            self._shrink_count += 1
            self._grow_count = 0
        elif decode_depth < self.target * 0.5:
            self._grow_count += 1
            self._shrink_count = 0
        else:
            self._shrink_count = 0
            self._grow_count = 0

        if self._shrink_count >= self.hold:
            self.chunk = max(self.min, self.chunk // 2)
            self._shrink_count = 0
        elif self._grow_count >= self.hold:
            self.chunk = min(self.max, self.chunk * 2)
            self._grow_count = 0

        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).debug(
                "ChunkSizeController step=%d depth=%d chunk=%d shrink=%d grow=%d",
                self._step_count, decode_depth, self.chunk,
                self._shrink_count, self._grow_count,
            )
        return self.chunk

'''

if "_shrink_count" not in src:
    src = src.replace(
        "class Scheduler(SchedulerInterface):",
        CHUNK_CLASS + "class Scheduler(SchedulerInterface):",
    )
    changed = True
    print("Patch 1 applied: ChunkSizeController class with hysteresis")
else:
    print("Patch 1 already present: ChunkSizeController class")

# ── Patch 2: __init__ wiring ─────────────────────────────────────────────────
INIT_PATCH = """
        # Dynamic chunk size controller (DYNAMIC_CHUNK=1 to enable).
        _dynamic_chunk = os.getenv("DYNAMIC_CHUNK", "0").strip() == "1"
        if _dynamic_chunk:
            _target = int(os.getenv("DYNAMIC_CHUNK_TARGET", "8"))
            _min = int(os.getenv("DYNAMIC_CHUNK_MIN", "256"))
            _hold = int(os.getenv("DYNAMIC_CHUNK_HOLD", "3"))
            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(
                min_tokens=_min,
                max_tokens=self.max_num_scheduled_tokens,
                target=_target,
                hold=_hold,
            )
            logger.info(
                "Dynamic chunk size enabled: min=%d max=%d target=%d hold=%d",
                _min, self.max_num_scheduled_tokens, _target, _hold,
            )
        else:
            self._chunk_ctrl = None
        _prefix_reorder = os.getenv("PREFIX_REORDER", "0").strip() == "1"
        self._prefix_reorder = _prefix_reorder
        self._aging_threshold_ms = float(os.getenv("AGING_THRESHOLD_MS", "inf"))
        self._aging_alpha = float(os.getenv("AGING_ALPHA", "0.3"))
        if _prefix_reorder:
            logger.info(
                "Prefix-aware request reordering enabled (alpha=%.2f)",
                self._aging_alpha,
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

# ── Patch 2b: _aging_alpha in __init__ (upgrade from pre-soft-aging) ─────────
_OLD_THRESH = '        self._aging_threshold_ms = float(os.getenv("AGING_THRESHOLD_MS", "inf"))'
_ALPHA_LINE = '        self._aging_alpha = float(os.getenv("AGING_ALPHA", "0.3"))'

if "_aging_alpha" not in src:
    if _OLD_THRESH not in src:
        print("ERROR: Patch 2b anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(_OLD_THRESH, _OLD_THRESH + "\n" + _ALPHA_LINE)
    changed = True
    print("Patch 2b applied: _aging_alpha")
else:
    print("Patch 2b already present: _aging_alpha")

# ── Patch 2c: _hold in __init__ (upgrade from pre-hysteresis) ────────────────
_OLD_MIN_LINE = '            _min = int(os.getenv("DYNAMIC_CHUNK_MIN", "256"))'
_HOLD_LINES = (
    '            _hold = int(os.getenv("DYNAMIC_CHUNK_HOLD", "3"))'
)
_OLD_CTRL_CALL = (
    '            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(\n'
    '                min_tokens=_min,\n'
    '                max_tokens=self.max_num_scheduled_tokens,\n'
    '                target=_target,\n'
    '            )'
)
_NEW_CTRL_CALL = (
    '            _hold = int(os.getenv("DYNAMIC_CHUNK_HOLD", "3"))\n'
    '            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(\n'
    '                min_tokens=_min,\n'
    '                max_tokens=self.max_num_scheduled_tokens,\n'
    '                target=_target,\n'
    '                hold=_hold,\n'
    '            )'
)

if "DYNAMIC_CHUNK_HOLD" not in src:
    if _OLD_CTRL_CALL not in src:
        print("ERROR: Patch 2c anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(_OLD_CTRL_CALL, _NEW_CTRL_CALL)
    changed = True
    print("Patch 2c applied: _hold / DYNAMIC_CHUNK_HOLD")
else:
    print("Patch 2c already present: DYNAMIC_CHUNK_HOLD")

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

# ── Patch 3b: warm-first soft-aging reorder block in schedule() ──────────────
REORDER_BLOCK = """            if self._prefix_reorder and self.waiting and hasattr(self.waiting, 'extendleft'):
                import time as _time
                import math as _math
                _now = _time.time()
                _alpha = self._aging_alpha
                def _soft_priority(req):
                    total = req.num_prompt_tokens
                    if total == 0:
                        hit_ratio = 0.0
                    elif req.num_computed_tokens > 0:
                        hit_ratio = req.num_computed_tokens / total
                    else:
                        _, cached = self.kv_cache_manager.get_computed_blocks(req)
                        hit_ratio = cached / total
                    wait_s = _now - req.arrival_time
                    return hit_ratio + _alpha * _math.log1p(wait_s)
                _sorted = sorted(self.waiting, key=_soft_priority, reverse=True)
                self.waiting.clear()
                self.waiting.extend(_sorted)
"""

if "_soft_priority" not in src:
    anchor_reorder = "            step_skipped_waiting = create_request_queue(self.policy)\n"
    if anchor_reorder not in src:
        print("ERROR: Patch 3b anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(anchor_reorder, anchor_reorder + REORDER_BLOCK)
    changed = True
    print("Patch 3b applied: soft-aging reorder block in schedule()")
else:
    print("Patch 3b already present: soft-aging reorder block")

if changed:
    SCHED.write_text(src)
    print(f"\nWrote patched scheduler to {SCHED}")
else:
    print("\nAll patches already present — no changes written.")
