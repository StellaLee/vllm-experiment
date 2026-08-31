#!/usr/bin/env python3
"""Static-budget + decode-reserve, no whale detection at all -- the ablation the
whale-aware-budget result needs to justify the dynamic gating over.

Sarathi-Serve's own design couples a fixed aggregate budget with explicit decode-phase
prioritization (decode iterations scheduled first, prefill chunks fill the remainder).
vLLM's actual V1 scheduler does not carry that distinction through (self.running is walked
in plain arrival order, confirmed present in vLLM main as of commit 94a54f5, two days after
v0.28.0). This patch restores exactly that piece -- decode-reserve reordering -- on top of a
BUDGET THAT NEVER CHANGES, to isolate what decode-reserve alone buys without whale-aware's
state-dependent gating (hotpatch_whale_aware_budget_v2.py) on top of it.

If this arm already matches whale-aware-v2 on tail protection (p99/p99.9/max), the dynamic
gating adds nothing beyond restoring Sarathi-Serve's own design intent, and the paper's
contribution shrinks to the fidelity-gap finding alone. If it doesn't -- if fixing the budget
at the protective value costs the same whales/mono throughput/TTFT whale-aware avoids -- that
is direct evidence the state-dependent gating is doing real, separate work, not just
restoring the fidelity gap.

Env:
  STATIC_RESERVE=1          enable
  STATIC_RESERVE_BUDGET=512 token_budget, held fixed for the entire run
  STATIC_RESERVE_TRACE=path.csv  optional: wall_s,token_budget per schedule() call (sanity
                             check only -- the second column should be constant)

Idempotent. Two edits: a budget-fix block (same anchor as the other budget controllers) and
the decode-reserve reorder block, same reorder logic/anchor as oracle_budget_reserve and
whale_aware_budget_v2.
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
)

ANCHOR_TOPUP = "        token_budget = self.max_num_scheduled_tokens\n"

BUDGET_BLOCK = ANCHOR_TOPUP + '''\
        if os.getenv("STATIC_RESERVE"):
            token_budget = int(os.getenv("STATIC_RESERVE_BUDGET", "512"))
            _sr_trace = os.getenv("STATIC_RESERVE_TRACE")
            if _sr_trace:
                import time as _sr_time
                with open(_sr_trace, "a") as _sr_f:
                    _sr_f.write(f"{_sr_time.monotonic()},{token_budget}\\n")
'''

ANCHOR_REORDER = (
    "        req_index = 0\n"
    "        while req_index < len(self.running) and token_budget > 0:\n"
)

REORDER_BLOCK = (
    "        if os.getenv(\"STATIC_RESERVE\"):\n"
    "            self.running = sorted(\n"
    "                self.running,\n"
    "                key=lambda _sr_r: 0 if _sr_r.num_computed_tokens >= _sr_r.num_prompt_tokens else 1,\n"
    "            )\n"
) + ANCHOR_REORDER

EDITS = [
    (ANCHOR_TOPUP, BUDGET_BLOCK, "static budget fix"),
    (ANCHOR_REORDER, REORDER_BLOCK, "decode-reserve reorder"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if BUDGET_BLOCK in src and REORDER_BLOCK in src:
        print("Static-reserve controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing "
            "static_reserve -- these are standalone alternatives, not meant to be "
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
    print(f"Static-reserve controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
