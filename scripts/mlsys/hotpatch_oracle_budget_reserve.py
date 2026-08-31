#!/usr/bin/env python3
"""Oracle budget switch + decode-reserve fix -- the constructive test the necessity argument
predicts should work, closing the monopolization gap found in hotpatch_fairshare_budget.py
and (partially) in the plain hotpatch_oracle_budget.py.

Combines two pieces:
  1. Same phase-based token_budget switch as hotpatch_oracle_budget.py (16384 during LO,
     512 during HI, oracle-known wall-clock phase boundary).
  2. Decode-reserve: reorder self.running each round so decode-only requests (prefill
     already fully computed) are processed FIRST, before prefill-continuing ones. Because
     the admission loop drains token_budget in list order, this structurally guarantees
     decode gets served before any of the (possibly shrunk) budget goes to prefill
     continuations -- not by accident of arrival order (which is what made the plain
     oracle-budget arm work most of the time but not always), but unconditionally, every
     round. This does NOT touch the waiting-loop's admission logic or long_prefill_token_
     threshold at all -- pure reordering of an existing list, same admission code runs
     unmodified after it.

Does NOT fix the separate transition-boundary problem (budget snapping back to unrestricted
at a phase change, letting carried-over whale backlog run free) -- that failure mode is
orthogonal to monopolization and needs its own fix if it turns out to still matter here.

Env:
  ORACLE_BUDGET_RESERVE=1   enable
  ORACLE_LO_S=120           seconds of the "LO" phase per cycle
  ORACLE_HI_S=120           seconds of the "HI" phase per cycle
  ORACLE_LO_BUDGET=16384    token_budget during LO
  ORACLE_HI_BUDGET=512      token_budget during HI
  ORACLE_TRACE=path.csv     optional: wall_s,elapsed_s,budget per schedule() call

Idempotent. Two edits: the budget-switch block (same anchor/logic as
hotpatch_oracle_budget.py) and a reorder block right before the running-loop begins.
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


OTHER_CONTROLLER_TAGS = (
    "ADAPTIVE_LPT", "FAIRSHARE_LPT", "FAIRSHARE_BUDGET", "FAIRSHARE_BOTH",
    "FAIRSHARE_RESERVE", "DRR_LPT", "ORACLE_LPT", "ORACLE_BUDGET",
    "ORACLE_BUDGET_RESERVE",
)

ANCHOR_TOPUP = "        token_budget = self.max_num_scheduled_tokens\n"

BUDGET_BLOCK = ANCHOR_TOPUP + '''\
        if os.getenv("ORACLE_BUDGET_RESERVE"):
            import time as _obr_time
            if not hasattr(self, "_obr_t0"):
                self._obr_t0 = _obr_time.monotonic()
            _obr_lo_s = float(os.getenv("ORACLE_LO_S", "120"))
            _obr_hi_s = float(os.getenv("ORACLE_HI_S", "120"))
            _obr_lo_budget = int(os.getenv("ORACLE_LO_BUDGET", "16384"))
            _obr_hi_budget = int(os.getenv("ORACLE_HI_BUDGET", "512"))
            _obr_cycle = _obr_lo_s + _obr_hi_s
            _obr_elapsed = _obr_time.monotonic() - self._obr_t0
            _obr_pos = _obr_elapsed % _obr_cycle if _obr_cycle > 0 else 0.0
            token_budget = _obr_lo_budget if _obr_pos < _obr_lo_s else _obr_hi_budget
            _obr_trace = os.getenv("ORACLE_TRACE")
            if _obr_trace:
                with open(_obr_trace, "a") as _obr_f:
                    _obr_f.write(f"{_obr_time.monotonic()},{_obr_elapsed:.2f},{token_budget}\\n")
'''

ANCHOR_REORDER = (
    "        req_index = 0\n"
    "        while req_index < len(self.running) and token_budget > 0:\n"
)

REORDER_BLOCK = (
    "        if os.getenv(\"ORACLE_BUDGET_RESERVE\"):\n"
    "            self.running = sorted(\n"
    "                self.running,\n"
    "                key=lambda _obr_r: 0 if _obr_r.num_computed_tokens >= _obr_r.num_prompt_tokens else 1,\n"
    "            )\n"
) + ANCHOR_REORDER

EDITS = [
    (ANCHOR_TOPUP, BUDGET_BLOCK, "budget-switch"),
    (ANCHOR_REORDER, REORDER_BLOCK, "decode-reserve reorder"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if BUDGET_BLOCK in src and REORDER_BLOCK in src:
        print("Oracle-budget-reserve controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing "
            "oracle_budget_reserve -- these are standalone alternatives, not meant to be "
            "stacked.",
            file=sys.stderr,
        )
        return 1

    for anchor, block, label in EDITS:
        if anchor not in src:
            print(f"ERROR: anchor not found for {label}", file=sys.stderr)
            return 1
        src = src.replace(anchor, block, 1)

    sched.write_text(src)
    print(f"Oracle-budget-reserve controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
