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
(512), ADAPTIVE_LPT_OFF (0), ADAPTIVE_LPT_TRACE (optional CSV path: wall_s,pf_remaining,threshold,
running,in_congest,token_budget per step).

Congestion-aware budget degradation (see docs/superpowers/specs/2026-08-27-congestion-aware-
adaptive-lpt-design.md): the per-request cap alone leaves the step-wide token_budget uncapped for
everything else, so under heavy load a protected request's rounds get slow (backlog fills the rest
of the budget every round), extending its residency and compounding the backlog further. When
ADAPTIVE_LPT_CONGEST_FRAC > 0 (default 0 = feature off, byte-identical to the plain hysteresis
behavior), a second independent hysteresis gate watches len(self.running) / max_num_running_reqs
-- load as a FRACTION of configured concurrency, not an absolute count, so the gate is expressed
relative to the server's own --max-num-seqs and doesn't need recalibrating if that changes. While
both protect-mode and congest-mode are active, token_budget itself is shrunk to _alpt_protect (the
SAME value ADAPTIVE_LPT_PROTECT already uses for the per-request cap -- not a second free
parameter; both mechanisms share one "how aggressive" magnitude). The congest-mode exit floor is
fixed at half the entry fraction (not independently configurable), same asymmetric-hysteresis
spirit as the pf_remaining gate above (avoid flapping) without adding a second tunable number.
ADAPTIVE_LPT_CONGEST_FRAC is therefore the only new knob this extension introduces.
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

HYSTERESIS_BLOCK = ANCHOR + '''\
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

# Backward-compat alias: earlier versions of this module (and any external callers) referred to
# the hysteresis block as NEW_BLOCK.
NEW_BLOCK = HYSTERESIS_BLOCK

CONGEST_BLOCK = ANCHOR + '''\
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
            _alpt_congest_frac = float(os.getenv("ADAPTIVE_LPT_CONGEST_FRAC", "0"))
            _alpt_running = len(self.running)
            _alpt_cap = max(1, self.max_num_running_reqs)
            _alpt_load = _alpt_running / _alpt_cap
            _alpt_in_congest = False
            if _alpt_congest_frac > 0:
                _alpt_was_congest = getattr(self, "_alpt_in_congest", False)
                _alpt_c_enter = _alpt_load > _alpt_congest_frac
                _alpt_c_stay = _alpt_was_congest and _alpt_load > (_alpt_congest_frac / 2)
                _alpt_in_congest = _alpt_c_enter or _alpt_c_stay
                self._alpt_in_congest = _alpt_in_congest
                if _alpt_in_protect and _alpt_in_congest:
                    token_budget = min(token_budget, _alpt_protect)
            _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
            if _alpt_trace:
                import time as _alpt_time
                with open(_alpt_trace, "a") as _alpt_f:
                    _alpt_f.write(
                        f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr},"
                        f"{_alpt_running},{_alpt_load:.4f},{int(_alpt_in_congest)},{token_budget}\\n")
'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if CONGEST_BLOCK in src:
        print("adaptive-lpt controller (congestion-aware) already present -- no changes.")
        return 0
    if HYSTERESIS_BLOCK in src:
        src = src.replace(HYSTERESIS_BLOCK, CONGEST_BLOCK, 1)
        sched.write_text(src)
        print(f"adaptive-lpt controller upgraded to congestion-aware version in {sched}")
        return 0
    if OLD_BLOCK in src:
        src = src.replace(OLD_BLOCK, CONGEST_BLOCK, 1)
        sched.write_text(src)
        print(f"adaptive-lpt controller upgraded (pre-hysteresis -> congestion-aware) in {sched}")
        return 0
    if ANCHOR not in src:
        print("ERROR: token_budget anchor not found", file=sys.stderr)
        return 1
    src = src.replace(ANCHOR, CONGEST_BLOCK, 1)
    sched.write_text(src)
    print(f"adaptive-lpt controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
