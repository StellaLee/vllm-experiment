#!/usr/bin/env python3
"""Fair-share per-request prefill-length threshold: no gate, no hysteresis, no calibrated
constant. Every step, cap any long request's admission to a fair share of the step's total
budget, split evenly among everything currently competing for it:

    threshold = max(1, max_num_scheduled_tokens // (1 + len(self.running)))

Standalone alternative to scripts/mlsys/hotpatch_adaptive_lpt.py -- not a layer on top of it, a
genuinely simpler design (see docs/superpowers/specs/2026-08-27-fairshare-lpt-design.md). Needs
zero calibrated numbers: it's a structural read of how many things are currently competing for
the budget, not a threshold anyone chose by fitting to a workload.

Why this needs no separate "is a long prefill present" gate: vLLM's own threshold semantics
(scheduler.py, both the running-loop and waiting-loop admission sites) only ever act when
`0 < threshold < num_new_tokens` -- a request already under `threshold` is untouched. So applying
this unconditionally, every step, to every request is a no-op for anything short; only requests
that actually need more tokens than their fair share are ever capped. The gate falls out for
free.

Why this needs no hysteresis: it's a continuous function of directly-observed state each step
(not a binary switch based on a threshold crossing), so there's nothing to flap between.

Idempotent. One edit: insert a per-step assignment right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line, so it runs every step regardless of whether
DYNAMIC_CHUNK/_chunk_ctrl or ADAPTIVE_LPT is active (though only one controller should be
installed on a given box at a time -- see scripts/mlsys/README or the plan doc for how to revert
to a pristine scheduler.py between controller experiments).

Env: FAIRSHARE_LPT (set to enable), FAIRSHARE_LPT_TRACE (optional CSV path:
wall_s,running,threshold per step -- track this to confirm the threshold is actually varying
across the run rather than stalling at the ceiling (16384, i.e. never engaging) or the floor
(127, i.e. always maxed out).
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


ANCHOR = "        token_budget = self.max_num_scheduled_tokens\n"

FAIRSHARE_BLOCK = ANCHOR + '''\
        if os.getenv("FAIRSHARE_LPT"):
            _fs_running = len(self.running)
            _fs_thr = max(1, self.max_num_scheduled_tokens // (1 + _fs_running))
            self.scheduler_config.long_prefill_token_threshold = _fs_thr
            _fs_trace = os.getenv("FAIRSHARE_LPT_TRACE")
            if _fs_trace:
                import time as _fs_time
                with open(_fs_trace, "a") as _fs_f:
                    _fs_f.write(f"{_fs_time.monotonic()},{_fs_running},{_fs_thr}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if FAIRSHARE_BLOCK in src:
        print("fairshare-lpt controller already present -- no changes.")
        return 0
    if "ADAPTIVE_LPT" in src or "FAIRSHARE_LPT" in src:
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first (see the .pristine backup, or "
            "hotpatch_adaptive_lpt's own anchor/end-marker slice) before installing "
            "fairshare-lpt -- these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, FAIRSHARE_BLOCK, 1)
    sched.write_text(src)
    print(f"fairshare-lpt controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
