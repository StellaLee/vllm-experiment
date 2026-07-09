#!/usr/bin/env python3
"""Upgrade the already-patched scheduler: replace the hard aging cliff with
a continuous soft-priority score: hit_ratio + alpha * log1p(wait_seconds).

Safe to re-run — skips hunks already present.
"""
import sys
from pathlib import Path

SCHED = Path("/root/miniconda3/lib/python3.10/site-packages/vllm/v1/core/sched/scheduler.py")
src = SCHED.read_text()
changed = False

# ── 1. Add self._aging_alpha to __init__ ────────────────────────────────────
OLD_AGING_LINE = '        self._aging_threshold_ms = float(os.getenv("AGING_THRESHOLD_MS", "inf"))'
NEW_AGING_LINES = (
    '        self._aging_threshold_ms = float(os.getenv("AGING_THRESHOLD_MS", "inf"))\n'
    '        self._aging_alpha = float(os.getenv("AGING_ALPHA", "0.3"))'
)

if "_aging_alpha" not in src:
    if OLD_AGING_LINE not in src:
        print("ERROR: _aging_threshold_ms anchor not found", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_AGING_LINE, NEW_AGING_LINES)
    changed = True
    print("Step 1 applied: _aging_alpha added to __init__")
else:
    print("Step 1 already present: _aging_alpha")

# ── 2. Replace hard-cliff REORDER_BLOCK with soft priority ──────────────────
OLD_REORDER = (
    "            if self._prefix_reorder and self.waiting and hasattr(self.waiting, 'extendleft'):\n"
    "                import time as _time\n"
    "                _now = _time.time()\n"
    "                _thresh_s = self._aging_threshold_ms / 1000.0\n"
    "                def _hit_ratio(req):\n"
    "                    total = req.num_prompt_tokens\n"
    "                    if total == 0:\n"
    "                        return 0.0\n"
    "                    if req.num_computed_tokens > 0:\n"
    "                        cached = req.num_computed_tokens\n"
    "                    else:\n"
    "                        _, cached = self.kv_cache_manager.get_computed_blocks(req)\n"
    "                    return cached / total\n"
    "                _aged = sorted(\n"
    "                    [r for r in self.waiting if (_now - r.arrival_time) >= _thresh_s],\n"
    "                    key=lambda r: r.arrival_time,\n"
    "                )\n"
    "                _fresh = sorted(\n"
    "                    [r for r in self.waiting if (_now - r.arrival_time) < _thresh_s],\n"
    "                    key=_hit_ratio, reverse=True,\n"
    "                )\n"
    "                self.waiting.clear()\n"
    "                self.waiting.extend(_aged + _fresh)\n"
)

NEW_REORDER = (
    "            if self._prefix_reorder and self.waiting and hasattr(self.waiting, 'extendleft'):\n"
    "                import time as _time\n"
    "                import math as _math\n"
    "                _now = _time.time()\n"
    "                _alpha = self._aging_alpha\n"
    "                def _soft_priority(req):\n"
    "                    total = req.num_prompt_tokens\n"
    "                    if total == 0:\n"
    "                        hit_ratio = 0.0\n"
    "                    elif req.num_computed_tokens > 0:\n"
    "                        hit_ratio = req.num_computed_tokens / total\n"
    "                    else:\n"
    "                        _, cached = self.kv_cache_manager.get_computed_blocks(req)\n"
    "                        hit_ratio = cached / total\n"
    "                    wait_s = _now - req.arrival_time\n"
    "                    return hit_ratio + _alpha * _math.log1p(wait_s)\n"
    "                _sorted = sorted(self.waiting, key=_soft_priority, reverse=True)\n"
    "                self.waiting.clear()\n"
    "                self.waiting.extend(_sorted)\n"
)

if "_soft_priority" not in src:
    if OLD_REORDER not in src:
        print("ERROR: old REORDER_BLOCK not found — scheduler may be at unexpected version", file=sys.stderr)
        print("Searching for nearby anchor...", file=sys.stderr)
        if "self._prefix_reorder and self.waiting" in src:
            print("Found prefix_reorder check but block differs — manual inspection needed", file=sys.stderr)
        sys.exit(1)
    src = src.replace(OLD_REORDER, NEW_REORDER)
    changed = True
    print("Step 2 applied: soft priority sort replaces hard aging cliff")
else:
    print("Step 2 already present: soft priority sort")

if changed:
    SCHED.write_text(src)
    print(f"\nWrote {SCHED}")
    print("Soft aging upgrade complete. Restart vLLM to take effect.")
    print("New env var: AGING_ALPHA (float, default 0.3)")
    print("  score = hit_ratio + AGING_ALPHA * log(1 + wait_seconds)")
else:
    print("\nNo changes needed — already at soft aging version.")
