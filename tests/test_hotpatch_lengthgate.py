#!/usr/bin/env python3
"""Offline test of the lengthgate patcher: apply to a scheduler FIXTURE (the anchors the real
scheduler contains), verify all four edits land and re-applying is a no-op. No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import hotpatch_lengthgate as H

FIXTURE = '''\
class ChunkSizeController:
    def step(self, decode_depth: int, last_tokens=None) -> int:
        if self.mode == "hslo":
            return self._step_hslo(decode_depth, last_tokens)
        if self.mode == "slo":
            return self._step_slo(decode_depth)
        return self._step_depth(decode_depth)


class Scheduler(SchedulerInterface):
    def schedule(self):
        token_budget = self.max_num_scheduled_tokens
        if self._chunk_ctrl is not None:
            _decode_depth = sum(
                1 for r in self.running if not r.is_prefill_chunk
            )
            token_budget = self._chunk_ctrl.step(
                _decode_depth, getattr(self, "_ff_last_tokens", None))
'''


def _apply(src):
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(src)
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    return rc, open(p).read()


def test_patch_applies_all_four_edits():
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'def step(self, decode_depth: int, last_tokens=None, pf_remaining=None)' in out
    assert 'if self.mode == "lengthgate":' in out
    assert 'def _step_lengthgate' in out
    assert '_pf_remaining' in out and '_decode_depth, getattr(self, "_ff_last_tokens", None), _pf_remaining)' in out


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()                       # second apply on already-patched file
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1   # no further change


def _run():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for f in fns:
        f(); print(f"ok {f.__name__}")
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run()
