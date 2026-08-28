#!/usr/bin/env python3
"""Offline test of the DRR-lite patcher: verify all 5 edit sites land, in the
right order, idempotent, and refuses to stack on another controller. No
vLLM/box needed. Fixture mirrors the real scheduler.py structure closely
enough to exercise the exact-string anchors used by the patcher."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_drr_lpt as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens

        if self._chunk_ctrl is not None:
            token_budget = self._chunk_ctrl.step(0, None)
        if self._pause_state == PauseState.PAUSED_ALL:
            token_budget = 0

        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]
            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            if num_new_tokens == 0:
                req_index += 1
                continue

            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

        while (self.waiting or self.skipped_waiting) and token_budget > 0:
            if True:
                if load_kv_async:
                    num_new_tokens = 0
                else:
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    num_new_tokens = min(num_new_tokens, token_budget)
                    assert num_new_tokens > 0

                request = request_queue.pop_request()
                self.running.append(request)
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_all_five_sites_land_in_order():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    idxs = [out.index(text) for text in [
        'if os.getenv("DRR_LPT"):\n            _drr_quantum',
        '_drr_rid = request.request_id\n                _drr_cap = self._drr_deficit.get(_drr_rid, 0)\n                if num_new_tokens > _drr_cap:',
        'if os.getenv("DRR_LPT"):\n                self._drr_deficit[request_id] = max(\n                    0, self._drr_deficit.get(request_id, 0) - num_new_tokens)\n            req_index += 1',
        '_drr_cap = max(1, self._drr_deficit.get(_drr_rid, 0))',
        'if os.getenv("DRR_LPT"):\n                    self._drr_deficit[request_id] = max(\n                        0, self._drr_deficit.get(request_id, 0) - num_new_tokens)\n                request.status = RequestStatus.RUNNING',
    ]]
    assert idxs == sorted(idxs), "edit sites landed out of order"


def test_patch_aggregate_budget_untouched():
    # regression guard: none of the 4 prior failed variants' aggregate-budget-shrink
    # patterns should be reintroduced
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert "token_budget = _fsr_reserve" not in out
    assert "token_budget = max(1," not in out
    assert "token_budget = _alpt_protect" not in out


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
        '        if os.getenv("FAIRSHARE_RESERVE"):\n'
        "            pass\n"))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 1
    assert 'if os.getenv("DRR_LPT"):\n            _drr_quantum' not in out


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
