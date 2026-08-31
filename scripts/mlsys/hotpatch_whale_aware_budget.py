#!/usr/bin/env python3
"""Whale-aware budget + decode-reserve: a REAL (non-oracle) controller that replaces the
oracle's wall-clock phase knowledge with a directly-observable signal -- whether any
long-prefill ("whale") request is currently mid-prefill.

    n_whales_active = count of requests in self.running still mid-prefill
                       (num_computed_tokens < num_prompt_tokens) with
                       num_prompt_tokens > WHALE_TOK (default 4000, this project's
                       standing whale-size convention)
    token_budget = HI_BUDGET (512) if n_whales_active > 0 else LO_BUDGET (16384)

Unlike every earlier reactive controller this session (fairshare_lpt/budget/both/reserve,
drr_lpt), which all used len(self.running) -- a proxy that's high both when legitimate
decode traffic is busy AND when whales are present, unable to tell the two apart -- this
uses the actual causal variable directly. No schedule foreknowledge needed (no LO_S/HI_S
tuned to a known cycle length): it reacts to whether a whale is truly present, so it should
also naturally avoid the oracle+reserve combination's remaining transition-boundary gap (the
oracle releases the cap on a fixed clock regardless of whether a straggling whale is still
mid-prefill; this only releases once every whale is truly done).

Combined with the same decode-reserve fix validated in hotpatch_oracle_budget_reserve.py:
reorder self.running each round so decode-only requests are always processed before prefill
continuations, guaranteeing their share of the (possibly shrunk) budget structurally.

Env:
  WHALE_AWARE_BUDGET=1        enable
  WHALE_AWARE_WHALE_TOK=4000  prompt-token threshold to count as a "whale"
  WHALE_AWARE_LO_BUDGET=16384 token_budget when no whale is active
  WHALE_AWARE_HI_BUDGET=512   token_budget when >=1 whale is active
  WHALE_AWARE_TRACE=path.csv  optional: wall_s,n_whales_active,token_budget per schedule() call

Idempotent. Two edits: the whale-presence budget block (same anchor as
hotpatch_oracle_budget.py) and the decode-reserve reorder block (same anchor/logic as
hotpatch_oracle_budget_reserve.py).
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
    (ANCHOR_TOPUP, BUDGET_BLOCK, "whale-aware budget"),
    (ANCHOR_REORDER, REORDER_BLOCK, "decode-reserve reorder"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if BUDGET_BLOCK in src and REORDER_BLOCK in src:
        print("Whale-aware-budget controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing "
            "whale_aware_budget -- these are standalone alternatives, not meant to be "
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
    print(f"Whale-aware-budget controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
