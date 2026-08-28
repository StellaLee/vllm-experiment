#!/usr/bin/env python3
"""Fair-share applied to BOTH the aggregate step budget and the per-request threshold, with the
same single formula and no calibrated constant:

    fair_share = max(1, max_num_scheduled_tokens // (1 + len(self.running)))
    token_budget = fair_share
    long_prefill_token_threshold = fair_share

Why both, not just one: hotpatch_fairshare_budget.py (aggregate-only) was found -- via direct
trace + record correlation on WRATE=0.5 under the widened-whale/max-tokens=1024 workload, see
docs/superpowers/specs/2026-08-27-fairshare-lpt-design.md -- to reproduce lengthgate's original
starvation bug (findings/2026-07-22-lengthgate-and-per-request-cap.md): shrinking the aggregate
budget without also bounding any single request's own slice of it lets one long-prefill request
monopolize the entire (now much smaller) round budget every step, starving every other admission
for as many consecutive rounds as it takes for its remaining need to drop below the shrunk
budget -- confirmed directly: the worst-hit records in that run were ordinary short requests
(prompt_tokens 69-939) going silent for 11-12 SECONDS mid-decode, with no single slow round (max
249ms) and no preemptions logged, which only fits a monopolized-budget explanation.

hotpatch_fairshare_lpt.py (per-request-threshold-only, the original attempt) has the opposite gap:
capping only the flagged request's slice leaves aggregate round cost uncapped, so backlog from
everything ELSE can still balloon round time under load -- this was the original WRATE=2.0 failure.

Applying the SAME formula to both closes both gaps at once, still with a single arithmetic line
and no new calibrated number: the aggregate is bounded (like static-512, but self-loosening when
idle), AND no single request -- long-prefill or otherwise -- can take more than its fair share of
that bounded round, so it can't monopolize and starve bystanders.

Idempotent. One edit: insert a per-step assignment right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line.

Env: FAIRSHARE_BOTH (set to enable), FAIRSHARE_BOTH_TRACE (optional CSV path:
wall_s,running,fair_share per step -- token_budget and long_prefill_token_threshold are always
set to the same value, so one column covers both).
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

FAIRSHARE_BOTH_BLOCK = ANCHOR + '''\
        if os.getenv("FAIRSHARE_BOTH"):
            _fsx_running = len(self.running)
            _fsx_share = max(1, self.max_num_scheduled_tokens // (1 + _fsx_running))
            token_budget = _fsx_share
            self.scheduler_config.long_prefill_token_threshold = _fsx_share
            _fsx_trace = os.getenv("FAIRSHARE_BOTH_TRACE")
            if _fsx_trace:
                import time as _fsx_time
                with open(_fsx_trace, "a") as _fsx_f:
                    _fsx_f.write(f"{_fsx_time.monotonic()},{_fsx_running},{_fsx_share}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if FAIRSHARE_BOTH_BLOCK in src:
        print("fairshare-both controller already present -- no changes.")
        return 0
    if any(tag in src for tag in ("ADAPTIVE_LPT", "FAIRSHARE_LPT", "FAIRSHARE_BUDGET", "FAIRSHARE_BOTH")):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing fairshare-both -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, FAIRSHARE_BOTH_BLOCK, 1)
    sched.write_text(src)
    print(f"fairshare-both controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
