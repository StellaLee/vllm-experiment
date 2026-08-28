#!/usr/bin/env python3
"""Fair-share STEP-WIDE BUDGET (not per-request threshold): no gate, no hysteresis, no
calibrated constant. Every step, shrink the step's total token budget to a fair share, split
evenly among everything currently competing for it:

    token_budget = max(1, max_num_scheduled_tokens // (1 + len(self.running)))

This is the budget-level counterpart to scripts/mlsys/hotpatch_fairshare_lpt.py, which applies
the identical formula to `long_prefill_token_threshold` instead -- and was found (see
docs/superpowers/specs/2026-08-27-fairshare-lpt-design.md, WRATE=0.5/1.0/2.0 validation) to lose
to static-512 on every axis at every rate. Root cause, not a workload artifact: capping only the
FLAGGED long request's own slice leaves every OTHER admission in the same round (decode tokens
for the whole running population, plus any other short prefills) completely uncapped, so round
wall-time -- which the project's own iteration-heartbeat model (step_time ~= db + alpha*B) says is
driven by TOTAL tokens processed that round -- can still balloon from backlog alone, independent
of how well-chosen the per-request cap is. static-512 wins under load specifically because it
bounds AGGREGATE round cost, not because "512" is a magic number for this workload.

This patch applies the SAME self-normalizing formula to the right target: the step-wide budget
itself, closing that gap directly, while (unlike static-512's constant 512) self-loosening to the
full budget when running=0 (no needless throughput tax when idle) and self-tightening under load.

Known risk from project history, NOT assumed safe: an earlier controller (lengthgate,
hotpatch_lengthgate.py) also shrank the step-wide budget, gated on prompt length rather than load,
and had a starvation bug -- a whale hogging a fully-shrunk budget while bystanders got zero
admission room, dumping a backlog into one expensive catch-up step. This design is continuous and
load-proportional rather than a binary length-triggered toggle, which may avoid that specific
failure mode, but this must be checked via trace forensics (FAIRSHARE_BUDGET_TRACE), the same
discipline that caught lengthgate's bug, not assumed from a clean-looking summary table.

Idempotent. One edit: insert a per-step reassignment right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line.

Env: FAIRSHARE_BUDGET (set to enable), FAIRSHARE_BUDGET_TRACE (optional CSV path:
wall_s,running,token_budget per step).
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

FAIRSHARE_BUDGET_BLOCK = ANCHOR + '''\
        if os.getenv("FAIRSHARE_BUDGET"):
            _fsb_running = len(self.running)
            token_budget = max(1, self.max_num_scheduled_tokens // (1 + _fsb_running))
            _fsb_trace = os.getenv("FAIRSHARE_BUDGET_TRACE")
            if _fsb_trace:
                import time as _fsb_time
                with open(_fsb_trace, "a") as _fsb_f:
                    _fsb_f.write(f"{_fsb_time.monotonic()},{_fsb_running},{token_budget}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if FAIRSHARE_BUDGET_BLOCK in src:
        print("fairshare-budget controller already present -- no changes.")
        return 0
    if "ADAPTIVE_LPT" in src or "FAIRSHARE_LPT" in src or "FAIRSHARE_BUDGET" in src:
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing fairshare-budget -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, FAIRSHARE_BUDGET_BLOCK, 1)
    sched.write_text(src)
    print(f"fairshare-budget controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
