#!/usr/bin/env python3
"""Upgrade ChunkSizeController to use hysteresis: only change chunk size after
HOLD consecutive steps above/below the threshold. Prevents rapid oscillation.

New env var: DYNAMIC_CHUNK_HOLD (int, default 3)
  score = require HOLD consecutive steps before growing/shrinking.

Safe to re-run.
"""
import sys
from pathlib import Path

SCHED = Path("/root/miniconda3/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py")
src = SCHED.read_text()
changed = False

# ── 1. Replace ChunkSizeController class ────────────────────────────────────
OLD_CLASS = '''class ChunkSizeController:
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
            logger.debug(
                "ChunkSizeController step=%d decode_depth=%d chunk=%d",
                self._step_count, decode_depth, self.chunk,
            )
        return self.chunk'''

NEW_CLASS = '''class ChunkSizeController:
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
            logger.debug(
                "ChunkSizeController step=%d depth=%d chunk=%d shrink=%d grow=%d",
                self._step_count, decode_depth, self.chunk,
                self._shrink_count, self._grow_count,
            )
        return self.chunk'''

if "_shrink_count" not in src:
    if OLD_CLASS not in src:
        print("ERROR: old ChunkSizeController not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_CLASS, NEW_CLASS)
    changed = True
    print("Step 1 applied: ChunkSizeController upgraded with hysteresis")
else:
    print("Step 1 already present: hysteresis counts")

# ── 2. Add hold parameter to ChunkSizeController instantiation in __init__ ──
OLD_INIT = (
    "            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(\n"
    "                min_tokens=_min,\n"
    "                max_tokens=self.max_num_scheduled_tokens,\n"
    "                target=_target,\n"
    "            )"
)
NEW_INIT = (
    "            _hold = int(os.getenv(\"DYNAMIC_CHUNK_HOLD\", \"3\"))\n"
    "            self._chunk_ctrl: ChunkSizeController | None = ChunkSizeController(\n"
    "                min_tokens=_min,\n"
    "                max_tokens=self.max_num_scheduled_tokens,\n"
    "                target=_target,\n"
    "                hold=_hold,\n"
    "            )"
)

if "DYNAMIC_CHUNK_HOLD" not in src:
    if OLD_INIT not in src:
        print("ERROR: ChunkSizeController instantiation anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_INIT, NEW_INIT)
    changed = True
    print("Step 2 applied: hold parameter wired to DYNAMIC_CHUNK_HOLD env var")
else:
    print("Step 2 already present: DYNAMIC_CHUNK_HOLD")

if changed:
    SCHED.write_text(src)
    print(f"\nWrote {SCHED}")
    print("Hysteresis upgrade complete. Restart vLLM to take effect.")
    print("New env var: DYNAMIC_CHUNK_HOLD (int, default 3)")
    print("  chunk only changes after HOLD consecutive steps above/below threshold")
else:
    print("\nNo changes needed — already at hysteresis version.")
