#!/usr/bin/env python3
"""Offline test of the fairshare-reserve patcher: verify token_budget is structurally larger
than long_prefill_token_threshold (the actual fix for the monopolization bug), idempotent, and
refuses to stack on another controller. No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_fairshare_reserve as H

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


def test_patch_token_budget_strictly_exceeds_share():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'if os.getenv("FAIRSHARE_RESERVE"):' in out
    # the actual fix: token_budget = reserve + share, NOT token_budget = share (the bug in
    # hotpatch_fairshare_both.py) -- so a single request capped at `share` can never consume the
    # whole round when reserve > 0.
    assert 'token_budget = _fsr_reserve + _fsr_share' in out
    assert 'self.scheduler_config.long_prefill_token_threshold = _fsr_share' in out
    # regression guard: must NOT reintroduce the both.py bug of budget == threshold
    assert 'token_budget = _fsr_share\n' not in out
    idx_anchor = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_gate = out.index('if os.getenv("FAIRSHARE_RESERVE"):')
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
        '        if os.getenv("FAIRSHARE_BOTH"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'if os.getenv("FAIRSHARE_RESERVE"):' not in out


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
