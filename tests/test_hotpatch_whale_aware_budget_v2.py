#!/usr/bin/env python3
"""Offline test of the whale-aware-budget v2 patcher: verify both edits land in order,
idempotent, refuses to stack, and (the actual fix over v1) checks self.waiting too, not
just self.running. No vLLM/box needed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_whale_aware_budget_v2 as H

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
    idx_budget = out.index('if os.getenv("WHALE_AWARE_BUDGET"):\n            _wab_whale_tok')
    idx_chunk_ctrl = out.index("if self._chunk_ctrl is not None:")
    idx_reorder = out.index('self.running = sorted(')
    idx_loop = out.index("while req_index < len(self.running) and token_budget > 0:")
    assert idx_anchor1 < idx_budget < idx_chunk_ctrl < idx_reorder < idx_loop


def test_counts_waiting_requests_too():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert "for _wab_w in self.waiting" in out
    assert "_wab_w.num_prompt_tokens > _wab_whale_tok" in out
    # regression guard: v1's gap was counting self.running only -- must now also count waiting
    budget_block = out.split('if os.getenv("WHALE_AWARE_BUDGET"):')[1].split("token_budget = _wab_hi_budget")[0]
    assert "self.running" in budget_block and "self.waiting" in budget_block


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
        '        if os.getenv("ORACLE_BUDGET_RESERVE"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'WHALE_AWARE_BUDGET"):\n            _wab_whale_tok' not in out


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
