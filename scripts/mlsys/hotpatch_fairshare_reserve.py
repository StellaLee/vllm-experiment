#!/usr/bin/env python3
"""Fair-share with an explicit decode reserve, fixing the monopolization bug found in
hotpatch_fairshare_both.py. Still zero calibrated constants:

    reserve = len(self.running)                                   # ~1 tok/seq for decode, cheap
    share   = max(1, (max_num_scheduled_tokens - reserve) // (1 + len(self.running)))
    token_budget = reserve + share
    long_prefill_token_threshold = share

Why hotpatch_fairshare_both.py failed (confirmed on WRATE=0.5 under the widened-whale/
max-tokens=1024 workload, see docs/superpowers/specs/2026-08-27-fairshare-lpt-design.md): it set
BOTH token_budget and long_prefill_token_threshold to the identical fair_share value. That does
not prevent monopolization -- if a single long request is allowed to take up to `fair_share`
tokens, and the entire round's budget is ALSO only `fair_share` tokens, then that one request
taking its maximum allowed slice consumes the whole round by itself, leaving nothing for anyone
else. Same starvation mechanism as hotpatch_fairshare_budget.py's original bug, just relabeled at
a smaller scale -- result was WORSE (W:max 15994.9ms) than the original aggregate-only bug
(11925.9ms).

The fix here makes the aggregate STRUCTURALLY larger than what any single request can claim:
`token_budget` always equals `reserve + share`, which is strictly greater than `share` whenever
`reserve > 0` (i.e. whenever anything else is running) -- so a single long request maxing out its
own threshold can never consume the entire round; `reserve` tokens are always left over. The
reserve amount (one token per currently-running sequence) is not a tuned constant either -- it is
exactly enough to cover one decode step for everyone already in the system, which is the specific
class of request (already-decoding, not the long request) that was found starving in both prior
failures (the worst-hit records in the fairshare_budget WRATE=0.5 run were short, already-decoding
requests going silent for 11-12 seconds, not whales).

Idempotent. One edit: insert a per-step block right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line.

Env: FAIRSHARE_RESERVE (set to enable), FAIRSHARE_RESERVE_TRACE (optional CSV path:
wall_s,running,reserve,share,token_budget per step -- track this to check for a repeat of the
monopolization signature: any already-running sequence going silent for far longer than a single
round's wall-time would take).
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

FAIRSHARE_RESERVE_BLOCK = ANCHOR + '''\
        if os.getenv("FAIRSHARE_RESERVE"):
            _fsr_running = len(self.running)
            _fsr_reserve = _fsr_running
            _fsr_share = max(1, (self.max_num_scheduled_tokens - _fsr_reserve) // (1 + _fsr_running))
            token_budget = _fsr_reserve + _fsr_share
            self.scheduler_config.long_prefill_token_threshold = _fsr_share
            _fsr_trace = os.getenv("FAIRSHARE_RESERVE_TRACE")
            if _fsr_trace:
                import time as _fsr_time
                with open(_fsr_trace, "a") as _fsr_f:
                    _fsr_f.write(
                        f"{_fsr_time.monotonic()},{_fsr_running},{_fsr_reserve},"
                        f"{_fsr_share},{token_budget}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if FAIRSHARE_RESERVE_BLOCK in src:
        print("fairshare-reserve controller already present -- no changes.")
        return 0
    if any(tag in src for tag in ("ADAPTIVE_LPT", "FAIRSHARE_LPT", "FAIRSHARE_BUDGET", "FAIRSHARE_BOTH", "FAIRSHARE_RESERVE")):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing fairshare-reserve -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, FAIRSHARE_RESERVE_BLOCK, 1)
    sched.write_text(src)
    print(f"fairshare-reserve controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
