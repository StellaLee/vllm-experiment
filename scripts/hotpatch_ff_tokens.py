#!/usr/bin/env python3
"""Feedforward v2 wiring: give the ChunkSizeController the ACTUAL number of tokens the scheduler
placed in the previous step, so tau = dt/tokens uses real tokens (not the granted budget). Two
edits to the live scheduler.py:
  1. record `total_num_scheduled_tokens` into `self._ff_last_tokens` after it is computed;
  2. pass it into `self._chunk_ctrl.step(_decode_depth, self._ff_last_tokens)`.
Run AFTER patch_scheduler.py + hotpatch_slo_tail.py (v2). Idempotent / re-runnable."""
import os, sys
from pathlib import Path


def _find_sched() -> Path:
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


SCHED = _find_sched()
if not SCHED.exists():
    print(f"ERROR: {SCHED} not found", file=sys.stderr); sys.exit(1)
src = SCHED.read_text()

if "_ff_last_tokens" in src:
    print("feedforward v2 token wiring already present — no changes."); sys.exit(0)

# Edit 1: pass the recorded actual token count into the controller.
OLD_CALL = "            token_budget = self._chunk_ctrl.step(_decode_depth)\n"
NEW_CALL = ("            token_budget = self._chunk_ctrl.step(\n"
            "                _decode_depth, getattr(self, \"_ff_last_tokens\", None))\n")
if OLD_CALL not in src:
    print("ERROR: controller call anchor not found (run patch_scheduler.py first)", file=sys.stderr); sys.exit(1)
src = src.replace(OLD_CALL, NEW_CALL, 1)

# Edit 2: record the actual scheduled-token total right after it is asserted.
OLD_REC = ("        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())\n"
           "        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens\n")
NEW_REC = (OLD_REC +
           "        if getattr(self, \"_chunk_ctrl\", None) is not None:\n"
           "            self._ff_last_tokens = total_num_scheduled_tokens\n")
if OLD_REC not in src:
    print("ERROR: total_num_scheduled_tokens anchor not found", file=sys.stderr); sys.exit(1)
src = src.replace(OLD_REC, NEW_REC, 1)

SCHED.write_text(src)
print(f"feedforward v2 token wiring installed in {SCHED}")
