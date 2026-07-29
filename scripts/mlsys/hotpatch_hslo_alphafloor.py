#!/usr/bin/env python3
"""Targeted, idempotent upgrade of an already-installed _step_hslo to add the ALPHA HARDWARE FLOOR.

The first hslo install underestimated alpha on noisy small-prefill steps: a near-zero alpha reading
made budget=(SLO-db)/alpha rail to the ceiling (16384) even with ~13 decoders present -> a whole
whale prefilled in one iteration -> multi-second TBT-max leak (diagnosed: 359/361 ceiling steps had
decode_depth>0). Flooring the online alpha at the offline hardware cost (DYNAMIC_CHUNK_ALPHA_MIN,
ms/token) blocks that downward excursion, so budget <= (SLO-db)/alpha_hw and predicted_iter <= SLO
BY CONSTRUCTION. Two in-place edits to the installed _step_hslo; no-op if alpha_hw already present.
"""
import os
import sys
from pathlib import Path


def _find_sched() -> Path:
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


INIT_OLD = (
    '            self._alpha_min = int(os.getenv("DYNAMIC_CHUNK_ALPHA_MIN_PREFILL", "256"))\n'
    '            self._hslo_init = True\n'
)
INIT_NEW = (
    '            self._alpha_min = int(os.getenv("DYNAMIC_CHUNK_ALPHA_MIN_PREFILL", "256"))\n'
    '            self._alpha_hw = float(os.getenv("DYNAMIC_CHUNK_ALPHA_MIN", "0.0")) / 1000.0\n'
    '            self._hslo_init = True\n'
)
BUDGET_OLD = "                self.chunk = int(max(self.min, min(self.max, headroom / self._alpha)))\n"
BUDGET_NEW = "                self.chunk = int(max(self.min, min(self.max, headroom / max(self._alpha, self._alpha_hw))))\n"


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if "def _step_hslo" not in src:
        print("ERROR: _step_hslo not installed; run hotpatch_hslo.py first", file=sys.stderr)
        return 1
    if "self._alpha_hw" in src:
        print("alpha floor already present -- no changes.")
        return 0
    if INIT_OLD not in src or BUDGET_OLD not in src:
        print("ERROR: expected _step_hslo anchors not found (method modified?)", file=sys.stderr)
        return 1
    src = src.replace(INIT_OLD, INIT_NEW, 1).replace(BUDGET_OLD, BUDGET_NEW, 1)
    sched.write_text(src)
    print(f"alpha floor installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
