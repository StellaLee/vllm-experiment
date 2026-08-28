#!/usr/bin/env python3
"""Offline test of the oracle-LPT patcher: verify the anchor edit lands, is idempotent,
refuses to stack, leaves token_budget untouched, and that the phase-selection math is
correct at boundaries. No vLLM/box needed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_oracle_lpt as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens

        if self._chunk_ctrl is not None:
            token_budget = self._chunk_ctrl.step(0, None)
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


def test_patch_lands_and_only_sets_threshold():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'if os.getenv("ORACLE_LPT"):' in out
    assert "self.scheduler_config.long_prefill_token_threshold = _oracle_thresh" in out
    # regression guard: must not touch token_budget at all (the lesson from 4 prior failures)
    assert "token_budget = _oracle" not in out
    assert "token_budget -=" not in out
    idx_anchor = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_oracle = out.index('if os.getenv("ORACLE_LPT"):')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    assert idx_anchor < idx_oracle < idx_chunk_ctrl


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1


def test_patch_refuses_to_stack_on_other_controller():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(FIXTURE.replace(
        "        token_budget = self.max_num_scheduled_tokens\n",
        "        token_budget = self.max_num_scheduled_tokens\n"
        '        if os.getenv("DRR_LPT"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'if os.getenv("ORACLE_LPT"):' not in out


def test_patch_missing_anchor_errors():
    rc, out = _apply("class Scheduler(SchedulerInterface):\n    def schedule(self):\n        pass\n")
    assert rc == 1


def _phase_threshold(elapsed, lo_s=120.0, hi_s=120.0, lo_thresh=0, hi_thresh=512):
    """Pure-python mirror of the patch's phase-selection formula, for boundary testing."""
    cycle = lo_s + hi_s
    pos = elapsed % cycle if cycle > 0 else 0.0
    return lo_thresh if pos < lo_s else hi_thresh


def test_phase_selection_boundaries():
    assert _phase_threshold(0.0) == 0
    assert _phase_threshold(119.9) == 0
    assert _phase_threshold(120.0) == 512
    assert _phase_threshold(239.9) == 512
    assert _phase_threshold(240.0) == 0          # wraps to next cycle's LO
    assert _phase_threshold(240.0 + 120.0) == 512  # second cycle's HI
    assert _phase_threshold(3 * 240.0 + 1.0) == 0  # third cycle wrap


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
