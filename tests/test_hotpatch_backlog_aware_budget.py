#!/usr/bin/env python3
"""Offline test of the backlog-aware-budget patcher: verify both edits land in order,
idempotent, refuses to stack, and that the budget block checks BOTH whale presence AND
pending backlog (the fix over whale-aware-v2, which only checked whale presence). No
vLLM/box needed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_backlog_aware_budget as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens

        if self._chunk_ctrl is not None:
            token_budget = self._chunk_ctrl.step(0, None)
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_both_edits_land_in_order():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    idx_anchor1 = out.index("token_budget = self.max_num_scheduled_tokens")
    idx_budget = out.index('if os.getenv("BACKLOG_AWARE_BUDGET"):\n            _bab_whale_tok')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    idx_reorder = out.index('self.running = sorted(')
    idx_loop = out.index("while req_index < len(self.running) and token_budget > 0:")
    assert idx_anchor1 < idx_budget < idx_chunk_ctrl < idx_reorder < idx_loop


def test_relaxes_only_when_whale_absent_and_backlog_low():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    budget_block = out.split('if os.getenv("BACKLOG_AWARE_BUDGET"):')[1].split("_bab_trace = ")[0]
    # regression guard: must check BOTH conditions, not whale presence alone (that was v2's gap)
    assert "_bab_n_whales > 0 or _bab_pending > _bab_backlog_thresh" in budget_block
    assert "self.waiting" in budget_block  # backlog counts waiting queue too, not just running


def test_pending_backlog_excludes_decode_only_computes_remaining_need():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert "_bab_r.num_prompt_tokens - _bab_r.num_computed_tokens" in out
    assert "_bab_r.num_computed_tokens < _bab_r.num_prompt_tokens" in out


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
        '        if os.getenv("WHALE_AWARE_BUDGET"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'BACKLOG_AWARE_BUDGET"):\n            _bab_whale_tok' not in out


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
