#!/usr/bin/env python3
"""Add CHUNK_MODE=lengthgate to vLLM's ChunkSizeController: blast a large prefill budget by default,
shrink to a small one whenever a LONG prompt is prefilling AND decoders are present. Length-driven,
never db-driven -- the freeze it protects against exists at any decode depth (no deep batch needed).

Idempotent. Four edits to scheduler.py:
  1. step() signature: add pf_remaining=None
  2. dispatch: route mode=="lengthgate" to _step_lengthgate
  3. method: insert _step_lengthgate before `class Scheduler(SchedulerInterface):`
  4. call site: compute _pf_remaining (max prefill-tokens-remaining over running prefill-chunks +
     waiting) and pass it to step(), so a queued whale is caught BEFORE it is admitted.

Env: LENGTHGATE_THRESHOLD (tok, default 4096), LENGTHGATE_PROTECT (512), LENGTHGATE_BLAST (16384).
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


SIG_ANCHOR = "    def step(self, decode_depth: int, last_tokens=None) -> int:\n"
SIG_NEW = "    def step(self, decode_depth: int, last_tokens=None, pf_remaining=None) -> int:\n"

DISPATCH_ANCHOR = (
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)
DISPATCH_NEW = (
    '        if self.mode == "lengthgate":\n'
    '            return self._step_lengthgate(decode_depth, pf_remaining)\n'
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)

METHOD_ANCHOR = "\nclass Scheduler(SchedulerInterface):\n"
METHOD = '''
    def _step_lengthgate(self, decode_depth: int, pf_remaining=None) -> int:
        # Prompt-length gate. Blast a large chunk by default (packs many short prefills / fast whale
        # TTFT); shrink to PROTECT whenever a long prompt is prefilling -- gated on LENGTH ALONE, not
        # decode_depth. Unlike a db-driven controller, protecting costs this controller nothing when
        # no one is decoding: slicing only delays the whale's OWN TTFT a little, never anyone else's.
        # Gating on depth>0 (as hslo does) is a liability here, not a feature: a whale that lands in a
        # momentary depth==0 lull would get blasted unsliced, freezing anyone who arrives DURING that
        # multi-second window -- exactly the failure this controller exists to avoid. Reacts in time
        # because pf_remaining includes WAITING requests -> a queued whale is caught before admission.
        thr = int(os.getenv("LENGTHGATE_THRESHOLD", "4096"))
        protect = int(os.getenv("LENGTHGATE_PROTECT", "512"))
        blast = int(os.getenv("LENGTHGATE_BLAST", "16384"))
        pf = int(pf_remaining or 0)
        if pf > thr:
            self.chunk = int(max(self.min, min(self.max, protect)))
        else:
            self.chunk = int(max(self.min, min(self.max, blast)))
        self._trace(decode_depth, float(pf))   # signal_ms column carries pf_remaining (diagnostics)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[lengthgate] step=%d depth=%d pf_remaining=%d chunk=%d",
                self._step_count, decode_depth, pf, self.chunk)
        return self.chunk

'''

CALL_ANCHOR = (
    "            token_budget = self._chunk_ctrl.step(\n"
    '                _decode_depth, getattr(self, "_ff_last_tokens", None))\n'
)
CALL_NEW = (
    "            _pf_remaining = 0\n"
    "            for _r in self.running:\n"
    "                if getattr(_r, 'is_prefill_chunk', False):\n"
    "                    _pf_remaining = max(_pf_remaining, _r.num_prompt_tokens - _r.num_computed_tokens)\n"
    "            for _r in self.waiting:\n"
    "                _pf_remaining = max(_pf_remaining, _r.num_prompt_tokens - _r.num_computed_tokens)\n"
    "            token_budget = self._chunk_ctrl.step(\n"
    '                _decode_depth, getattr(self, "_ff_last_tokens", None), _pf_remaining)\n'
)


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if "def _step_lengthgate" in src:
        print("lengthgate controller already present -- no changes.")
        return 0
    for name, anchor in (("step() signature", SIG_ANCHOR),
                         ("dispatch (mode=='slo')", DISPATCH_ANCHOR),
                         ("class Scheduler anchor", METHOD_ANCHOR),
                         ("controller.step call site", CALL_ANCHOR)):
        if anchor not in src:
            print(f"ERROR: {name} anchor not found", file=sys.stderr)
            return 1
    src = src.replace(SIG_ANCHOR, SIG_NEW, 1)
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_NEW, 1)
    src = src.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR, 1)
    src = src.replace(CALL_ANCHOR, CALL_NEW, 1)
    sched.write_text(src)
    print(f"lengthgate controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
