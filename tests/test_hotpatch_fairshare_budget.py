#!/usr/bin/env python3
"""Offline test of the fairshare-budget patcher: apply to a scheduler FIXTURE, verify the block
lands right after the anchor (and before _chunk_ctrl can override token_budget, so it composes
the same way every other controller in this file does), is idempotent, and refuses to stack on
top of a different already-installed controller patch. No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_fairshare_budget as H

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


def test_patch_applies_fairshare_budget_block():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'if os.getenv("FAIRSHARE_BUDGET"):' in out
    # reassigns token_budget itself (aggregate cap), not long_prefill_token_threshold
    assert 'token_budget = max(1, self.max_num_scheduled_tokens // (1 + _fsb_running))' in out
    assert 'long_prefill_token_threshold' not in out
    idx_anchor = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_gate = out.index('if os.getenv("FAIRSHARE_BUDGET"):')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    assert idx_anchor < idx_gate < idx_chunk_ctrl


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
        '        if os.getenv("FAIRSHARE_LPT"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'if os.getenv("FAIRSHARE_BUDGET"):' not in out


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
