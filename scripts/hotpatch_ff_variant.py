#!/usr/bin/env python3
"""Make the feedforward controller's tau-basis selectable by env CHUNK_FF_VARIANT:
  v2 (default) -> tau = dt / ACTUAL tokens processed last step (needs _ff_last_tokens wiring)
  v1           -> tau = dt / GRANTED budget last step (self._last_budget), ignore actual tokens
Lets us run both variants without flipping the scheduler wiring between servers. Idempotent."""
import os, sys
from pathlib import Path


def _find_sched() -> Path:
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


SCHED = _find_sched()
src = SCHED.read_text()
if "CHUNK_FF_VARIANT" in src:
    print("ff-variant toggle already present — no changes."); sys.exit(0)

OLD = "        T = last_tokens if (last_tokens is not None and last_tokens > 0) else self._last_budget\n"
NEW = ("        if os.getenv(\"CHUNK_FF_VARIANT\", \"v2\").strip().lower() == \"v1\":\n"
       "            T = self._last_budget if self._last_budget else last_tokens  # granted-budget basis\n"
       "        else:\n"
       "            T = last_tokens if (last_tokens is not None and last_tokens > 0) else self._last_budget\n")
if OLD not in src:
    print("ERROR: feedforward T= anchor not found (line ~216)", file=sys.stderr); sys.exit(1)
src = src.replace(OLD, NEW, 1)
SCHED.write_text(src)
print(f"ff-variant toggle installed in {SCHED}")
