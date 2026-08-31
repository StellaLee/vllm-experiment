#!/usr/bin/env python3
"""Pure-observation round tracer -- quantitative confirmation for the necessity argument
(2026-08-28): does a fixed per-request cap fail to bound tail latency because of (a)
aggregate round-volume inflating wall-clock step time (the established step_time ~=
decode_baseline + alpha*tokens_this_round model), (b) FCFS/list-order admission denial
(a running request gets zero new tokens this round, for one or more consecutive rounds),
or both?

Does NOT touch scheduling decisions -- inserts one read-only block right before
SchedulerOutput is constructed, where `total_num_scheduled_tokens` and
`num_scheduled_tokens` are already finalized for the round. Safe to run alongside the
NATIVE --long-prefill-token-threshold flag (no scheduler.py changes needed for that --
it's vLLM's own CLI-parsed config), or alongside mono (no threshold at all).

Logs, per round:
  wall_s, n_running, total_tokens_this_round, n_skipped, n_whales_active

  n_skipped        = requests in self.running that got 0 new tokens this round
                      (direct measurement of mechanism (b), not a proxy)
  total_tokens...   = aggregate tokens admitted this round across all requests
                      (the alpha-model input for mechanism (a))
  n_whales_active   = requests in self.running still mid-prefill with
                      prompt_tokens > ROUND_TRACE_WHALE_TOK (default 4000)

Env: ROUND_TRACE=path.csv (enable + output path), ROUND_TRACE_WHALE_TOK=4000 (optional).

Idempotent. One edit: insert a block right after the existing
`assert token_budget >= 0` line (unique anchor, right before SchedulerOutput is built).
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


ANCHOR = "        assert token_budget >= 0\n"

TRACE_BLOCK = ANCHOR + '''\
        _rtrace_path = os.getenv("ROUND_TRACE")
        if _rtrace_path:
            import time as _rtrace_time
            _rtrace_running_ids = {r.request_id for r in self.running}
            _rtrace_admitted_ids = set(num_scheduled_tokens.keys())
            _rtrace_skipped = len(_rtrace_running_ids - _rtrace_admitted_ids)
            _rtrace_whale_tok = int(os.getenv("ROUND_TRACE_WHALE_TOK", "4000"))
            _rtrace_whales_active = sum(
                1 for r in self.running
                if r.num_computed_tokens < r.num_prompt_tokens
                and r.num_prompt_tokens > _rtrace_whale_tok
            )
            with open(_rtrace_path, "a") as _rtrace_f:
                _rtrace_f.write(
                    f"{_rtrace_time.monotonic()},{len(self.running)},"
                    f"{total_num_scheduled_tokens},{_rtrace_skipped},{_rtrace_whales_active}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if TRACE_BLOCK in src:
        print("Round tracer already present -- no changes.")
        return 0

    if "ROUND_TRACE" in src and TRACE_BLOCK not in src:
        print("ERROR: a different ROUND_TRACE block already present (unexpected state).",
              file=sys.stderr)
        return 1

    if ANCHOR not in src:
        print("ERROR: 'assert token_budget >= 0' anchor not found", file=sys.stderr)
        return 1

    src = src.replace(ANCHOR, TRACE_BLOCK, 1)
    sched.write_text(src)
    print(f"Round tracer installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
