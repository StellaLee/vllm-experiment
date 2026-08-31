#!/usr/bin/env python3
"""Backlog-aware budget: fixes the regression found in whale-aware-v2 at rate=1.4, where
the dynamic controller was measurably WORSE than a permanently-tight static budget.

Diagnosis (2026-08-29): whale-aware-v2 relaxes to LO_BUDGET=16384 whenever no whale is
"currently present" -- but "no whale present" is a fact about arrivals, not about system
occupancy. At rate=1.0 that's harmless (nothing accumulates either way). At rate=1.4, full
relaxation during a quiet window lets more non-whale prefill requests get admitted
SIMULTANEOUSLY (loose budget in a single round pulls more of self.waiting's queue into
self.running than a tight budget would), inflating the population of still-mid-prefill
requests. When a whale then arrives and the budget snaps back to 512, that inflated
population of ALREADY-ADMITTED-BUT-UNFINISHED prefills, plus the whale, plus ongoing decode,
all compete for the same tightened round budget -- a pile-up static-reserve (which never
relaxes, so never inflates this population) never creates.

Fix: track PENDING PREFILL BACKLOG directly -- total remaining prefill tokens across every
request still mid-prefill in self.running, plus every request queued in self.waiting -- and
only relax the budget when whale presence AND pending backlog are BOTH low. Whale presence
alone is an incomplete proxy for whether relaxation is currently safe; this closes that gap
by checking the thing that actually matters (system occupancy), not just the trigger that
whale-aware-v2 used as a stand-in for it.

    n_whales_active   = (as in whale-aware-v2: whales in self.running + self.waiting)
    pending_backlog   = sum(remaining prefill tokens) over self.running (still mid-prefill)
                        and self.waiting (not yet admitted at all)
    token_budget = HI_BUDGET (512) if n_whales_active > 0 OR pending_backlog > BACKLOG_THRESH
                   else LO_BUDGET (16384)

Same decode-reserve reorder as whale-aware-v2, unchanged.

Env:
  BACKLOG_AWARE_BUDGET=1          enable
  BACKLOG_AWARE_WHALE_TOK=4000    prompt-token threshold to count as a "whale"
  BACKLOG_AWARE_LO_BUDGET=16384   token_budget when safe to relax
  BACKLOG_AWARE_HI_BUDGET=512     token_budget when a whale is active/imminent, or backlog high
  BACKLOG_AWARE_BACKLOG_THRESH=4000  pending prefill tokens above which relaxation is unsafe
                                  even with no whale present -- a first-guess default, not
                                  yet tuned; expect this to need a sweep of its own.
  BACKLOG_AWARE_TRACE=path.csv   optional: wall_s,n_whales_active,pending_backlog,token_budget

Idempotent. Two edits: the budget block (whale-presence OR backlog check) and the same
decode-reserve reorder block used by whale_aware_budget_v2.py, same anchors.
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
    "ORACLE_BUDGET_RESERVE", "WHALE_AWARE_BUDGET", "STATIC_RESERVE",
    "BACKLOG_AWARE_BUDGET",
)

ANCHOR_TOPUP = "        token_budget = self.max_num_scheduled_tokens\n"

BUDGET_BLOCK = ANCHOR_TOPUP + '''\
        if os.getenv("BACKLOG_AWARE_BUDGET"):
            _bab_whale_tok = int(os.getenv("BACKLOG_AWARE_WHALE_TOK", "4000"))
            _bab_backlog_thresh = int(os.getenv("BACKLOG_AWARE_BACKLOG_THRESH", "4000"))
            _bab_n_whales = sum(
                1 for _bab_r in self.running
                if _bab_r.num_computed_tokens < _bab_r.num_prompt_tokens
                and _bab_r.num_prompt_tokens > _bab_whale_tok
            ) + sum(
                1 for _bab_w in self.waiting
                if _bab_w.num_prompt_tokens > _bab_whale_tok
            )
            _bab_pending = sum(
                max(0, _bab_r.num_prompt_tokens - _bab_r.num_computed_tokens)
                for _bab_r in self.running
                if _bab_r.num_computed_tokens < _bab_r.num_prompt_tokens
            ) + sum(_bab_w.num_prompt_tokens for _bab_w in self.waiting)
            _bab_lo_budget = int(os.getenv("BACKLOG_AWARE_LO_BUDGET", "16384"))
            _bab_hi_budget = int(os.getenv("BACKLOG_AWARE_HI_BUDGET", "512"))
            if _bab_n_whales > 0 or _bab_pending > _bab_backlog_thresh:
                token_budget = _bab_hi_budget
            else:
                token_budget = _bab_lo_budget
            _bab_trace = os.getenv("BACKLOG_AWARE_TRACE")
            if _bab_trace:
                import time as _bab_time
                with open(_bab_trace, "a") as _bab_f:
                    _bab_f.write(
                        f"{_bab_time.monotonic()},{_bab_n_whales},{_bab_pending},{token_budget}\\n")
'''

ANCHOR_REORDER = (
    "        req_index = 0\n"
    "        while req_index < len(self.running) and token_budget > 0:\n"
)

REORDER_BLOCK = (
    "        if os.getenv(\"BACKLOG_AWARE_BUDGET\"):\n"
    "            self.running = sorted(\n"
    "                self.running,\n"
    "                key=lambda _bab_r2: 0 if _bab_r2.num_computed_tokens >= _bab_r2.num_prompt_tokens else 1,\n"
    "            )\n"
) + ANCHOR_REORDER

EDITS = [
    (ANCHOR_TOPUP, BUDGET_BLOCK, "backlog-aware budget"),
    (ANCHOR_REORDER, REORDER_BLOCK, "decode-reserve reorder"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if BUDGET_BLOCK in src and REORDER_BLOCK in src:
        print("Backlog-aware-budget controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing "
            "backlog_aware_budget -- these are standalone alternatives, not meant to be "
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
    print(f"Backlog-aware-budget controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
