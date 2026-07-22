#!/usr/bin/env python3
"""Make --long-prefill-token-threshold adaptive: off (0) by default, protect (512) whenever a
long prompt is prefilling or waiting. Independent of ChunkSizeController/DYNAMIC_CHUNK -- the
step-wide budget is never touched, only scheduler_config.long_prefill_token_threshold, which both
existing read sites (scheduler.py running-loop line ~736, waiting-loop line ~1032) read fresh
every step with no caching.

Idempotent. One edit: insert a per-step gate block right after the unconditional
`token_budget = self.max_num_scheduled_tokens` line, so it runs every step regardless of whether
DYNAMIC_CHUNK/_chunk_ctrl is active.

Env: ADAPTIVE_LPT (set to enable), ADAPTIVE_LPT_GATE (tok, default 4096) -- entry threshold,
ADAPTIVE_LPT_EXIT_GATE (tok, default 0) -- exit threshold (hysteresis: once protecting, stay
protected until pf_remaining drops to this floor, NOT just back under ADAPTIVE_LPT_GATE -- a
single shared entry/exit bar was found to relax protection ~4200-4500 tokens before a long
prompt actually finishes, dumping the uncapped remainder in one shot), ADAPTIVE_LPT_PROTECT
(512), ADAPTIVE_LPT_OFF (0), ADAPTIVE_LPT_TRACE (optional CSV path: wall_s,pf_remaining,threshold
per step).
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

OLD_BLOCK = ANCHOR + '''\
        if os.getenv("ADAPTIVE_LPT"):
            _alpt_pf = 0
            for _r in self.running:
                if getattr(_r, 'is_prefill_chunk', False):
                    _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            for _r in self.waiting:
                _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            _alpt_gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
            _alpt_protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
            _alpt_off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
            _alpt_thr = _alpt_protect if _alpt_pf > _alpt_gate else _alpt_off
            self.scheduler_config.long_prefill_token_threshold = _alpt_thr
            _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
            if _alpt_trace:
                import time as _alpt_time
                with open(_alpt_trace, "a") as _alpt_f:
                    _alpt_f.write(f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr}\\n")
'''

NEW_BLOCK = ANCHOR + '''\
        if os.getenv("ADAPTIVE_LPT"):
            _alpt_pf = 0
            for _r in self.running:
                if getattr(_r, 'is_prefill_chunk', False):
                    _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            for _r in self.waiting:
                _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
            _alpt_gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
            _alpt_exit_gate = int(os.getenv("ADAPTIVE_LPT_EXIT_GATE", "0"))
            _alpt_protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
            _alpt_off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
            _alpt_was_protect = getattr(self, "_alpt_in_protect", False)
            _alpt_enter = _alpt_pf > _alpt_gate
            _alpt_stay = _alpt_was_protect and _alpt_pf > _alpt_exit_gate
            _alpt_in_protect = _alpt_enter or _alpt_stay
            self._alpt_in_protect = _alpt_in_protect
            _alpt_thr = _alpt_protect if _alpt_in_protect else _alpt_off
            self.scheduler_config.long_prefill_token_threshold = _alpt_thr
            _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
            if _alpt_trace:
                import time as _alpt_time
                with open(_alpt_trace, "a") as _alpt_f:
                    _alpt_f.write(f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if NEW_BLOCK in src:
        print("adaptive-lpt controller (hysteresis) already present -- no changes.")
        return 0
    if OLD_BLOCK in src:
        src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
        sched.write_text(src)
        print(f"adaptive-lpt controller upgraded to hysteresis version in {sched}")
        return 0
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, NEW_BLOCK, 1)
    sched.write_text(src)
    print(f"adaptive-lpt controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
