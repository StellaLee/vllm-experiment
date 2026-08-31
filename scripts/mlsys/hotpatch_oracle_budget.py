#!/usr/bin/env python3
"""Oracle *aggregate budget* switch -- the token_budget analogue of hotpatch_oracle_lpt.py.

hotpatch_oracle_lpt.py tested oracle-switching the per-request cap
(long_prefill_token_threshold) between 0 (mono) and 512 (static-512) on a known wall-clock
phase boundary; it lost to both static baselines. This tests the other lever: switching the
*aggregate* round budget (max_num_batched_tokens, i.e. token_budget) instead, between 16384
(mono-equivalent) and 512 (matching the pre-existing "static-512 step-wide" arm from
057-adaptive-cap.tex's Table 1, which as a CONSTANT already achieves an even tighter tail
bound than the per-request cap -- W:max 261.4ms vs. 490.4ms -- at a steeper constant
throughput cost, S:TTFT 271ms, the worst of any arm there). Question: does oracle-gating that
steeper-but-tighter mechanism (pay it only during the known whale-heavy window) do better
than the per-request-cap oracle did?

Per the necessity argument (059-necessity.tex), we expect this to fail for a related but
distinct reason: with NO per-request cap, a single (or several) long-prefill continuation(s)
early in list order can still consume the entire shrunk budget in one round, since nothing
bounds any one request's SHARE of it -- this is structurally the same failure mode diagnosed
for hotpatch_fairshare_budget.py, just oracle-gated (fixed per-phase value) instead of
reactively recomputed. Testing this explicitly, rather than assuming, both closes an obvious
"why not just shrink the budget" reviewer question and is a further necessity-argument
corroboration if it fails, or a genuine finding if it doesn't.

Does NOT touch long_prefill_token_threshold at all -- orthogonal lever, safe to run under
native vLLM with no --long-prefill-token-threshold flag (mono's own default).

Env:
  ORACLE_BUDGET=1           enable
  ORACLE_LO_S=120           seconds of the "LO" (no/low whale) phase per cycle
  ORACLE_HI_S=120           seconds of the "HI" (whale-heavy) phase per cycle
  ORACLE_LO_BUDGET=16384    token_budget during LO (mono-equivalent)
  ORACLE_HI_BUDGET=512      token_budget during HI (matches the static-512-step-wide arm)
  ORACLE_TRACE=path.csv     optional: wall_s,elapsed_s,budget per schedule() call

Must be given the SAME LO_S/HI_S as the client's --phase-schedule so phase boundaries line
up, same as hotpatch_oracle_lpt.py.

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
    "FAIRSHARE_RESERVE", "DRR_LPT", "ORACLE_LPT", "ORACLE_BUDGET",
)

ANCHOR = "        token_budget = self.max_num_scheduled_tokens\n"

ORACLE_BUDGET_BLOCK = ANCHOR + '''\
        if os.getenv("ORACLE_BUDGET"):
            import time as _oraclebud_time
            if not hasattr(self, "_oraclebud_t0"):
                self._oraclebud_t0 = _oraclebud_time.monotonic()
            _oraclebud_lo_s = float(os.getenv("ORACLE_LO_S", "120"))
            _oraclebud_hi_s = float(os.getenv("ORACLE_HI_S", "120"))
            _oraclebud_lo_budget = int(os.getenv("ORACLE_LO_BUDGET", "16384"))
            _oraclebud_hi_budget = int(os.getenv("ORACLE_HI_BUDGET", "512"))
            _oraclebud_cycle = _oraclebud_lo_s + _oraclebud_hi_s
            _oraclebud_elapsed = _oraclebud_time.monotonic() - self._oraclebud_t0
            _oraclebud_pos = _oraclebud_elapsed % _oraclebud_cycle if _oraclebud_cycle > 0 else 0.0
            token_budget = _oraclebud_lo_budget if _oraclebud_pos < _oraclebud_lo_s else _oraclebud_hi_budget
            _oraclebud_trace = os.getenv("ORACLE_TRACE")
            if _oraclebud_trace:
                with open(_oraclebud_trace, "a") as _oraclebud_f:
                    _oraclebud_f.write(
                        f"{_oraclebud_time.monotonic()},{_oraclebud_elapsed:.2f},{token_budget}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if ORACLE_BUDGET_BLOCK in src:
        print("Oracle-budget controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing oracle_budget -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1

    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1

    src = src.replace(ANCHOR, ORACLE_BUDGET_BLOCK, 1)
    sched.write_text(src)
    print(f"Oracle-budget controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
