#!/usr/bin/env python3
"""DRR-lite: persistent per-request deficit credit replacing the static
long_prefill_token_threshold, while leaving the aggregate token_budget
completely untouched.

Motivation (2026-08-27, after 4 failed fair-share variants): every prior
attempt failed because it shrunk the AGGREGATE token_budget from instantaneous
`len(self.running)`, with no memory of who got shortchanged last round. A
request stuck deep in the scheduling order could starve for many consecutive
rounds even though it was fully active, because nothing tracked "how much has
this request actually gotten lately" -- decisions were remade from scratch,
context-free, every round.

Deficit Round Robin (classic fair-queuing algorithm, used for router bandwidth
sharing) fixes exactly this class of problem: every active request earns a
fixed QUANTUM of credit each round; it may be admitted up to
min(remaining_need, credit); unspent credit carries into the next round
(capped, so a request that goes uncontested for a while can't bank an
unbounded burst). A request that got shortchanged this round (aggregate
budget ran dry before reaching it) keeps its credit and is first in line to
use it once budget frees up.

    deficit[req] = min(deficit.get(req, 0) + QUANTUM, MAX_BANK)   # top-up, every round
    admit        = min(remaining_need, deficit[req])              # this round's cap
    deficit[req] -= admit                                         # spend

This v1 does NOT reorder self.running / self.waiting -- admission order is
still native list/FCFS order. It isolates one variable (memory/carryover)
from another (service order) rather than changing both at once. If this alone
doesn't help, that's itself informative: it would mean ordering, not memory,
is the missing piece.

QUANTUM defaults to 512 -- reusing the one constant already empirically
validated (static-512 is undefeated across every workload tested this
project), not re-derived from theory. This is deliberately not framed as a
zero-knob claim; it's a test of whether adding fairness memory ON TOP of the
known-good per-round unit improves on its memoryless FCFS version.
MAX_BANK defaults to 4x QUANTUM: bounds how large a single-round admission
burst can ever be (a request that banked credit across several contended
rounds), so DRR can't reintroduce an unbounded freeze -- the exact failure
mode chunking exists to prevent.

Env:
  DRR_LPT=1              enable
  DRR_QUANTUM=512         credit added per request per round (tokens)
  DRR_MAX_BANK=2048       cap on banked deficit (default 4x quantum)
  DRR_TRACE=path.csv      optional: wall_s,n_running,n_waiting,sum_deficit,max_deficit,n_at_bank_cap per round

Idempotent. Four edits: one bookkeeping block after the token_budget anchor,
a cap + spend pair in the running loop, a cap + spend pair in the waiting loop.
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
    "FAIRSHARE_RESERVE", "DRR_LPT",
)

ANCHOR_TOPUP = "        token_budget = self.max_num_scheduled_tokens\n"

BLOCK_TOPUP = ANCHOR_TOPUP + '''\
        if os.getenv("DRR_LPT"):
            _drr_quantum = int(os.getenv("DRR_QUANTUM", "512"))
            _drr_max_bank = int(os.getenv("DRR_MAX_BANK", str(_drr_quantum * 4)))
            if not hasattr(self, "_drr_deficit"):
                self._drr_deficit = {}
            _drr_active_ids = set()
            for _drr_req in self.running:
                _drr_rid = _drr_req.request_id
                _drr_active_ids.add(_drr_rid)
                self._drr_deficit[_drr_rid] = min(
                    self._drr_deficit.get(_drr_rid, 0) + _drr_quantum, _drr_max_bank)
            for _drr_req in self.waiting:
                _drr_rid = _drr_req.request_id
                _drr_active_ids.add(_drr_rid)
                self._drr_deficit[_drr_rid] = min(
                    self._drr_deficit.get(_drr_rid, 0) + _drr_quantum, _drr_max_bank)
            for _drr_rid in list(self._drr_deficit):
                if _drr_rid not in _drr_active_ids:
                    del self._drr_deficit[_drr_rid]
            _drr_trace = os.getenv("DRR_TRACE")
            if _drr_trace:
                import time as _drr_time
                _drr_vals = list(self._drr_deficit.values())
                _drr_at_cap = sum(1 for v in _drr_vals if v >= _drr_max_bank)
                with open(_drr_trace, "a") as _drr_f:
                    _drr_f.write(
                        f"{_drr_time.monotonic()},{len(self.running)},{len(self.waiting)},"
                        f"{sum(_drr_vals)},{max(_drr_vals) if _drr_vals else 0},{_drr_at_cap}\\n")
'''

ANCHOR_RUN_CAP = (
    "            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(num_new_tokens, token_budget)\n"
)

BLOCK_RUN_CAP = (
    "            if os.getenv(\"DRR_LPT\"):\n"
    "                _drr_rid = request.request_id\n"
    "                _drr_cap = self._drr_deficit.get(_drr_rid, 0)\n"
    "                if num_new_tokens > _drr_cap:\n"
    "                    num_new_tokens = _drr_cap\n"
    "            elif 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:\n"
    "                num_new_tokens = self.scheduler_config.long_prefill_token_threshold\n"
    "            num_new_tokens = min(num_new_tokens, token_budget)\n"
)

ANCHOR_RUN_SPEND = (
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            req_index += 1\n"
)

BLOCK_RUN_SPEND = (
    "            num_scheduled_tokens[request_id] = num_new_tokens\n"
    "            token_budget -= num_new_tokens\n"
    "            if os.getenv(\"DRR_LPT\"):\n"
    "                self._drr_deficit[request_id] = max(\n"
    "                    0, self._drr_deficit.get(request_id, 0) - num_new_tokens)\n"
    "            req_index += 1\n"
)

ANCHOR_WAIT_CAP = (
    "                    num_new_tokens = request.num_tokens - num_computed_tokens\n"
    "                    threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                    if 0 < threshold < num_new_tokens:\n"
    "                        num_new_tokens = threshold\n"
)

BLOCK_WAIT_CAP = (
    "                    num_new_tokens = request.num_tokens - num_computed_tokens\n"
    "                    if os.getenv(\"DRR_LPT\"):\n"
    "                        _drr_rid = request.request_id\n"
    "                        _drr_cap = max(1, self._drr_deficit.get(_drr_rid, 0))\n"
    "                        if num_new_tokens > _drr_cap:\n"
    "                            num_new_tokens = _drr_cap\n"
    "                    else:\n"
    "                        threshold = self.scheduler_config.long_prefill_token_threshold\n"
    "                        if 0 < threshold < num_new_tokens:\n"
    "                            num_new_tokens = threshold\n"
)

ANCHOR_WAIT_SPEND = (
    "                num_scheduled_tokens[request_id] = num_new_tokens\n"
    "                token_budget -= num_new_tokens\n"
    "                request.status = RequestStatus.RUNNING\n"
)

BLOCK_WAIT_SPEND = (
    "                num_scheduled_tokens[request_id] = num_new_tokens\n"
    "                token_budget -= num_new_tokens\n"
    "                if os.getenv(\"DRR_LPT\"):\n"
    "                    self._drr_deficit[request_id] = max(\n"
    "                        0, self._drr_deficit.get(request_id, 0) - num_new_tokens)\n"
    "                request.status = RequestStatus.RUNNING\n"
)

EDITS = [
    (ANCHOR_TOPUP, BLOCK_TOPUP, "topup"),
    (ANCHOR_RUN_CAP, BLOCK_RUN_CAP, "running-loop cap"),
    (ANCHOR_RUN_SPEND, BLOCK_RUN_SPEND, "running-loop spend"),
    (ANCHOR_WAIT_CAP, BLOCK_WAIT_CAP, "waiting-loop cap"),
    (ANCHOR_WAIT_SPEND, BLOCK_WAIT_SPEND, "waiting-loop spend"),
]


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()

    if 'if os.getenv("DRR_LPT"):' in src and BLOCK_TOPUP in src:
        print("DRR-lite controller already present -- no changes.")
        return 0

    if any(tag in src for tag in OTHER_CONTROLLER_TAGS):
        print(
            "ERROR: scheduler.py already has a different controller patch installed. "
            "Revert to a pristine scheduler.py first before installing drr_lpt -- "
            "these are standalone alternatives, not meant to be stacked.",
            file=sys.stderr,
        )
        return 1

    for anchor, block, label in EDITS:
        if anchor not in src:
            print(f"ERROR: anchor not found for {label}", file=sys.stderr)
            return 1
        src = src.replace(anchor, block, 1)

    sched.write_text(src)
    print(f"DRR-lite controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
