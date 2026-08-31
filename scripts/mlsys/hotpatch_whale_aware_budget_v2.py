#!/usr/bin/env python3
"""Whale-aware budget v2: closes the detection-lag gap found in
hotpatch_whale_aware_budget.py (v1).

v1 counted whales only in self.running -- computed at the top of schedule(), BEFORE this
round's new admissions happen. So the very first round a whale gets admitted, it isn't
counted yet, and the budget is still loose (16384) for that one round: every newly-arriving
whale got one free, uncapped first grab before protection kicked in the round after. That
cost v1 its worst-case metrics (p99.9, max) relative to the oracle, even though v1 beat the
oracle on mean/p99.

Fix: also count large prompts sitting in self.waiting, about to be admitted -- catching a
whale one round earlier, before its first chunk lands, not after.

    n_whales_active = (whales in self.running still mid-prefill)
                     + (requests in self.waiting with prompt_tokens > WHALE_TOK)
    token_budget = HI_BUDGET (512) if n_whales_active > 0 else LO_BUDGET (16384)

Everything else identical to v1: same decode-reserve reorder, same env var names/defaults
(so v1 and v2 sweep scripts are drop-in comparable).

Env:
  WHALE_AWARE_BUDGET=1        enable
  WHALE_AWARE_WHALE_TOK=4000  prompt-token threshold to count as a "whale"
  WHALE_AWARE_LO_BUDGET=16384 token_budget when no whale is active or about to be
  WHALE_AWARE_HI_BUDGET=512   token_budget when >=1 whale is active or about to be
  WHALE_AWARE_TRACE=path.csv  optional: wall_s,n_whales_active,token_budget per schedule() call

Idempotent. Two edits: the whale-presence budget block (extended to check self.waiting too)
and the decode-reserve reorder block, same anchors as v1.
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
    "ORACLE_BUDGET_RESERVE", "WHALE_AWARE_BUDGET",
)

ANCHOR_TOPUP = "        token_budget = self.max_num_scheduled_tokens\n"

BUDGET_BLOCK = ANCHOR_TOPUP + '''\
        if os.getenv("WHALE_AWARE_BUDGET"):
            _wab_whale_tok = int(os.getenv("WHALE_AWARE_WHALE_TOK", "4000"))
            _wab_n_whales = sum(
                1 for _wab_r in self.running
                if _wab_r.num_computed_tokens < _wab_r.num_prompt_tokens
                and _wab_r.num_prompt_tokens > _wab_whale_tok
            ) + sum(
                1 for _wab_w in self.waiting
                if _wab_w.num_prompt_tokens > _wab_whale_tok
            )
            _wab_lo_budget = int(os.getenv("WHALE_AWARE_LO_BUDGET", "16384"))
            _wab_hi_budget = int(os.getenv("WHALE_AWARE_HI_BUDGET", "512"))
            token_budget = _wab_hi_budget if _wab_n_whales > 0 else _wab_lo_budget
            _wab_trace = os.getenv("WHALE_AWARE_TRACE")
            if _wab_trace:
                import time as _wab_time
                with open(_wab_trace, "a") as _wab_f:
                    _wab_f.write(f"{_wab_time.monotonic()},{_wab_n_whales},{token_budget}\\n")
'''

ANCHOR_REORDER = (
    "        req_index = 0\n"
    "        while req_index < len(self.running) and token_budget > 0:\n"
)

REORDER_BLOCK = (
    "        if os.getenv(\"WHALE_AWARE_BUDGET\"):\n"
    "            self.running = sorted(\n"
    "                self.running,\n"
    "                key=lambda _wab_r2: 0 if _wab_r2.num_computed_tokens >= _wab_r2.num_prompt_tokens else 1,\n"
    "            )\n"
) + ANCHOR_REORDER

EDITS = [
    (ANCHOR_TOPUP, BUDGET_BLOCK, "whale-aware budget (v2, checks self.waiting too)"),
    (ANCHOR_REORDER, REORDER_BLOCK, "decode-reserve reorder"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if BUDGET_BLOCK in src and REORDER_BLOCK in src:
        print("Whale-aware-budget v2 controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing "
            "whale_aware_budget_v2 -- these are standalone alternatives, not meant to be "
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
    print(f"Whale-aware-budget v2 controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
