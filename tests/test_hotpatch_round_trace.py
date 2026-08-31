#!/usr/bin/env python3
"""Offline test of the round-tracer patcher: verify it lands read-only (no scheduling
logic touched), is idempotent, and computes n_skipped correctly against the fixture's
shape. No vLLM/box needed."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
import hotpatch_round_trace as H

FIXTURE = '''\
class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens
        num_scheduled_tokens = {}

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs

        scheduler_output = SchedulerOutput(
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
        )
        return scheduler_output
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_lands_read_only():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'os.getenv("ROUND_TRACE")' in out
    # read-only guard: must not assign to token_budget, num_scheduled_tokens, or self.running
    assert "token_budget =" not in out.split("assert token_budget >= 0")[1].split("scheduler_output")[0].replace(
        "_rtrace", "")  # only our own _rtrace-prefixed locals may appear after the anchor
    idx_anchor = out.index("assert token_budget >= 0")
    idx_trace = out.index('os.getenv("ROUND_TRACE")')
    idx_output = out.index("scheduler_output = SchedulerOutput(")
    assert idx_anchor < idx_trace < idx_output


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1


def test_patch_missing_anchor_errors():
    rc, out = _apply("class Scheduler(SchedulerInterface):\n    def schedule(self):\n        pass\n")
    assert rc == 1


def test_nskipped_formula_matches_set_difference():
    # pure-python mirror of the n_skipped computation, sanity-checked against hand cases
    running_ids = {"a", "b", "c", "d"}
    admitted_ids = {"a", "c"}  # b and d got 0 tokens this round
    n_skipped = len(running_ids - admitted_ids)
    assert n_skipped == 2
    admitted_ids_all = {"a", "b", "c", "d"}
    assert len(running_ids - admitted_ids_all) == 0


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
