#!/usr/bin/env python3
"""Oracle threshold switch: probes whether non-stationary load creates real headroom
for adaptive control, before building another real (reactive) controller.

Not a deployable mechanism -- it switches `long_prefill_token_threshold` on a known
wall-clock phase boundary (matching the client's --phase-schedule exactly), rather than
inferring the regime from observable state. This isolates the question "does adaptivity
have ANY theoretical ceiling above static under non-stationary load" from "can a real
controller find it" -- the same two questions kept getting conflated across this
session's five failed controllers.

Per 2026-08-28 decision: only `long_prefill_token_threshold` is switched.
`token_budget` (self.max_num_scheduled_tokens) is left completely untouched -- every
mechanism this session that shrunk the aggregate round budget caused starvation
(fairshare_budget, fairshare_both, fairshare_reserve); every mechanism that only
touched the per-request cap did not. That lesson doesn't change just because the
surrounding load is time-varying instead of stationary -- the starvation mechanism
(one request monopolizing a shrunk round) is a per-round, local phenomenon.

Because this only overwrites `self.scheduler_config.long_prefill_token_threshold`, the
existing native admission-cap code (untouched by this patch) picks it up automatically
at both existing call sites -- no new admission logic, minimal new surface area.

Env:
  ORACLE_LPT=1              enable
  ORACLE_LO_S=120           seconds of the "LO" (no/low whale) phase per cycle
  ORACLE_HI_S=120           seconds of the "HI" (whale-heavy) phase per cycle
  ORACLE_LO_THRESH=0        threshold during LO (0 = off, matches mono baseline)
  ORACLE_HI_THRESH=512      threshold during HI (matches static-512 baseline)
  ORACLE_TRACE=path.csv     optional: wall_s,elapsed_s,threshold per schedule() call

Must be given the SAME LO_S/HI_S as the client's --phase-schedule so phase boundaries
line up (the oracle's clock starts at its own first schedule() call, which coincides
with the first request's arrival -- effectively simultaneous with the client's first
send on an otherwise-idle server).

Idempotent. One edit: insert a block right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line.
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
    "FAIRSHARE_RESERVE", "DRR_LPT", "ORACLE_LPT",
)

ANCHOR = "        token_budget = self.max_num_scheduled_tokens\n"

ORACLE_BLOCK = ANCHOR + '''\
        if os.getenv("ORACLE_LPT"):
            import time as _oracle_time
            if not hasattr(self, "_oracle_t0"):
                self._oracle_t0 = _oracle_time.monotonic()
            _oracle_lo_s = float(os.getenv("ORACLE_LO_S", "120"))
            _oracle_hi_s = float(os.getenv("ORACLE_HI_S", "120"))
            _oracle_lo_thresh = int(os.getenv("ORACLE_LO_THRESH", "0"))
            _oracle_hi_thresh = int(os.getenv("ORACLE_HI_THRESH", "512"))
            _oracle_cycle = _oracle_lo_s + _oracle_hi_s
            _oracle_elapsed = _oracle_time.monotonic() - self._oracle_t0
            _oracle_pos = _oracle_elapsed % _oracle_cycle if _oracle_cycle > 0 else 0.0
            _oracle_thresh = _oracle_lo_thresh if _oracle_pos < _oracle_lo_s else _oracle_hi_thresh
            self.scheduler_config.long_prefill_token_threshold = _oracle_thresh
            _oracle_trace = os.getenv("ORACLE_TRACE")
            if _oracle_trace:
                with open(_oracle_trace, "a") as _oracle_f:
                    _oracle_f.write(
                        f"{_oracle_time.monotonic()},{_oracle_elapsed:.2f},{_oracle_thresh}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if ORACLE_BLOCK in src:
        print("Oracle-LPT controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing oracle_lpt -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1

    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1

    src = src.replace(ANCHOR, ORACLE_BLOCK, 1)
    sched.write_text(src)
    print(f"Oracle-LPT controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
