#!/usr/bin/env python3
"""Upgrade ChunkSizeController to support an SLO-feedback mode.

Adds CHUNK_MODE=slo: instead of controlling the prefill token budget on
decode-queue *depth* (the original bang-bang/hysteresis behaviour, kept as
CHUNK_MODE=depth), control it on *measured per-step iteration latency* vs a
TPOT SLO. This fixes the open-loop failure of depth mode, where deep-but-healthy
decode (25+ inflight) keeps reading "decode is deep" and over-shrinks the budget
to the floor, strangling prefill and collapsing TTFT.

  over SLO         -> halve chunk, but never below the stall-free floor (MIN)
  comfortably under -> grow chunk toward max (buys TTFT while decode has headroom)

The call site (ChunkSizeController(min,max,target,hold) and .step(decode_depth))
is UNCHANGED: SLO config is read from env inside the class, and per-step latency
is measured internally from the time between step() calls. So this only needs to
replace the class body.

New env vars (only used when CHUNK_MODE=slo):
  CHUNK_MODE            depth (default) | slo
  DYNAMIC_CHUNK_SLO_MS  target per-step (=per-token) latency in ms (default 50)
  DYNAMIC_CHUNK_STEP    additive-increase step in tokens (default = MIN)
  DYNAMIC_CHUNK_EMA     EMA weight for latency smoothing (default 0.3)
  DYNAMIC_CHUNK_MIN     stall-free floor; set 512 for the probe (base default 256)

Run patch_scheduler.py FIRST (this replaces the class it installs).
Safe to re-run. Restart vLLM to take effect.
"""
import re
import sys
from pathlib import Path

SCHED = Path("/root/miniconda3/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py")
if not SCHED.exists():
    print(f"ERROR: {SCHED} not found", file=sys.stderr)
    sys.exit(1)

src = SCHED.read_text()

if "_step_slo" in src and "CHUNK_MODE" in src:
    print("SLO controller already present — no changes.")
    sys.exit(0)

NEW_CLASS = '''class ChunkSizeController:
    """Chunk-prefill token-budget controller with two selectable modes.

    CHUNK_MODE=depth (default): bang-bang on decode-queue depth with hysteresis
        (original behaviour). Tune DYNAMIC_CHUNK_TARGET, DYNAMIC_CHUNK_HOLD.
    CHUNK_MODE=slo: AIMD on measured per-step iteration latency vs a TPOT SLO.
        Grows the prefill budget toward max when decode latency has headroom
        (buys TTFT); halves it toward the stall-free floor when latency exceeds
        the SLO (protects TPOT). Fixes the open-loop failure of depth mode,
        where deep-but-healthy decode over-shrinks the budget to the floor.
        Tune DYNAMIC_CHUNK_SLO_MS (default 50), DYNAMIC_CHUNK_STEP,
        DYNAMIC_CHUNK_EMA. Floor = DYNAMIC_CHUNK_MIN (set 512 for stall-free).

    Enable the controller with DYNAMIC_CHUNK=1.
    """

    def __init__(self, min_tokens: int, max_tokens: int, target: int,
                 hold: int = 3) -> None:
        import os
        self.chunk = max_tokens
        self.min = min_tokens
        self.max = max_tokens
        self.target = target
        self.hold = hold
        self._step_count = 0
        self._shrink_count = 0
        self._grow_count = 0
        # SLO-mode config (read here so the call site stays unchanged).
        self.mode = os.getenv("CHUNK_MODE", "depth").strip().lower()
        self.slo = float(os.getenv("DYNAMIC_CHUNK_SLO_MS", "50")) / 1000.0
        self.aimd_step = int(os.getenv("DYNAMIC_CHUNK_STEP", str(max(1, min_tokens))))
        self.ema = float(os.getenv("DYNAMIC_CHUNK_EMA", "0.3"))
        self._last_t = None
        self._lat = None
        import logging
        logging.getLogger(__name__).info(
            "ChunkSizeController mode=%s min=%d max=%d slo_ms=%.1f",
            self.mode, self.min, self.max, self.slo * 1000.0,
        )

    def step(self, decode_depth: int) -> int:
        self._step_count += 1
        if self.mode == "slo":
            return self._step_slo(decode_depth)
        return self._step_depth(decode_depth)

    def _step_depth(self, decode_depth: int) -> int:
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
        return self.chunk

    def _step_slo(self, decode_depth: int) -> int:
        import time
        now = time.monotonic()
        if self._last_t is not None:
            dt = now - self._last_t
            # Only fold in steps that reflect real decode work; drop idle gaps.
            if decode_depth > 0 and dt < 0.5:
                self._lat = dt if self._lat is None else (
                    self.ema * dt + (1.0 - self.ema) * self._lat)
        self._last_t = now
        if self._lat is not None:
            if self._lat > self.slo:
                # Over SLO: multiplicative back-off, never below the stall-free floor.
                self.chunk = max(self.min, self.chunk // 2)
            elif self._lat < self.slo * 0.75:
                # Latency headroom: additive growth toward max to buy TTFT.
                self.chunk = min(self.max, self.chunk + self.aimd_step)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).debug(
                "ChunkCtrl[slo] step=%d depth=%d lat_ms=%.1f chunk=%d",
                self._step_count, decode_depth,
                (self._lat or 0.0) * 1000.0, self.chunk,
            )
        return self.chunk

'''

# Replace the whole existing ChunkSizeController class (whichever prior version
# is installed) up to the blank line before `class Scheduler`. Regex is robust to
# depth-only vs hysteresis variants.
pattern = re.compile(
    r"class ChunkSizeController:.*?\n\n(?=class Scheduler\(SchedulerInterface\):)",
    re.DOTALL,
)
if not pattern.search(src):
    print("ERROR: ChunkSizeController class not found. Run patch_scheduler.py "
          "first to install the base controller.", file=sys.stderr)
    sys.exit(1)

src = pattern.sub(NEW_CLASS, src, count=1)
SCHED.write_text(src)
print(f"SLO controller installed in {SCHED}")
print("Enable with: DYNAMIC_CHUNK=1 CHUNK_MODE=slo DYNAMIC_CHUNK_MIN=512 "
      "DYNAMIC_CHUNK_SLO_MS=50")
print("Restart vLLM to take effect.")
