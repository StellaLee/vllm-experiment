#!/usr/bin/env python3
"""Offline test of the adaptive-lpt patcher: apply to a scheduler FIXTURE (the anchor line the
real scheduler contains), verify the gate block lands right after it and re-applying is a no-op.
No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import hotpatch_adaptive_lpt as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens

        if self._chunk_ctrl is not None:
            _decode_depth = sum(
                1 for r in self.running if not r.is_prefill_chunk
            )
            token_budget = self._chunk_ctrl.step(
                _decode_depth, getattr(self, "_ff_last_tokens", None))
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_applies_gate_block():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'if os.getenv("ADAPTIVE_LPT"):' in out
    assert 'self.scheduler_config.long_prefill_token_threshold = _alpt_thr' in out
    # hysteresis: entry and exit are governed by separate gates, with persisted state
    assert '_alpt_exit_gate = int(os.getenv("ADAPTIVE_LPT_EXIT_GATE", "0"))' in out
    assert '_alpt_was_protect = getattr(self, "_alpt_in_protect", False)' in out
    assert '_alpt_stay = _alpt_was_protect and _alpt_pf > _alpt_exit_gate' in out
    idx_anchor = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_gate = out.index('if os.getenv("ADAPTIVE_LPT"):')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    assert idx_anchor < idx_gate < idx_chunk_ctrl


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()                       # second apply on already-patched file
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1   # no further change


def test_patch_upgrades_old_block_to_hysteresis():
    # A file already patched with the pre-hysteresis (single shared entry/exit gate) block
    # should be upgraded in place to the new hysteresis block, not left alone or duplicated.
    rc0, out0 = _apply(FIXTURE)                       # fresh install (now the new block)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(FIXTURE.replace(H.ANCHOR, H.OLD_BLOCK, 1))
    rc1 = H.main()
    out1 = open(os.environ["VLLM_SCHED_PATH"]).read()
    assert rc1 == 0
    assert H.OLD_BLOCK not in out1
    assert H.NEW_BLOCK in out1
    assert out1.count('if os.getenv("ADAPTIVE_LPT"):') == 1   # no duplication


def test_patch_missing_anchor_errors():
    rc, out = _apply("class Scheduler(SchedulerInterface):\n    def schedule(self):\n        pass\n")
    assert rc == 1


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
