#!/usr/bin/env python3
"""Offline test of the adaptive-lpt patcher: apply to a scheduler FIXTURE (the anchor line the
real scheduler contains), verify the gate block lands right after it and re-applying is a no-op.
No vLLM/box needed."""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "mlsys"))
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


def test_patch_installs_congestion_block():
    # Fresh install lands the newest (congestion-aware) block directly.
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert 'ADAPTIVE_LPT_CONGEST_FRAC' in out
    assert '_alpt_in_congest' in out
    # load is a FRACTION of configured concurrency (self.max_num_running_reqs), not an absolute
    # running-sequence count, so the gate doesn't need recalibrating if --max-num-seqs changes.
    assert '_alpt_load = _alpt_running / _alpt_cap' in out
    idx_thr = out.index('self.scheduler_config.long_prefill_token_threshold = _alpt_thr')
    idx_congest = out.index('ADAPTIVE_LPT_CONGEST_FRAC')
    idx_trace = out.index('_alpt_trace = os.getenv')
    assert idx_thr < idx_congest < idx_trace


def test_patch_congest_frac_zero_is_noop_regression():
    # ADAPTIVE_LPT_CONGEST_FRAC unset/0 must be a full no-op vs. the plain hysteresis behavior:
    # token_budget is never touched, in_congest is always False. Verified structurally (the
    # guard `if _alpt_congest_frac > 0:` wraps every congestion-specific mutation) rather than by
    # executing the block, since that needs a live scheduler.
    rc, out = _apply(FIXTURE)
    assert rc == 0
    guard_idx = out.index("if _alpt_congest_frac > 0:")
    budget_mutation_idx = out.index("token_budget = min(token_budget, _alpt_protect)")
    assert guard_idx < budget_mutation_idx  # the mutation is nested inside the feature-flag guard


def test_patch_congestion_has_exactly_one_new_tunable():
    # Only ADAPTIVE_LPT_CONGEST_FRAC is a new free parameter -- the degraded budget reuses
    # _alpt_protect directly (no separate ADAPTIVE_LPT_CONGEST_BUDGET) and the exit floor is a
    # fixed ratio of the entry fraction (no separate ADAPTIVE_LPT_CONGEST_EXIT_FRAC), so there is
    # no second knob to fit to a particular workload/traffic pattern.
    rc, out = _apply(FIXTURE)
    assert rc == 0
    assert "ADAPTIVE_LPT_CONGEST_BUDGET" not in out
    assert "ADAPTIVE_LPT_CONGEST_EXIT_FRAC" not in out
    assert "_alpt_congest_frac / 2" in out


def test_patch_idempotent():
    rc1, out1 = _apply(FIXTURE)
    open(os.environ["VLLM_SCHED_PATH"], "w").write(out1)
    rc2 = H.main()                       # second apply on already-patched file
    assert rc2 == 0
    assert open(os.environ["VLLM_SCHED_PATH"]).read() == out1   # no further change


def test_patch_upgrades_old_block_to_congestion_aware():
    # A file already patched with the pre-hysteresis (single shared entry/exit gate) block
    # should be upgraded in place all the way to the congestion-aware block, not left at an
    # intermediate stage or duplicated.
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(FIXTURE.replace(H.ANCHOR, H.OLD_BLOCK, 1))
    os.environ["VLLM_SCHED_PATH"] = p
    rc1 = H.main()
    out1 = open(p).read()
    assert rc1 == 0
    assert H.OLD_BLOCK not in out1
    assert H.CONGEST_BLOCK in out1
    assert out1.count('if os.getenv("ADAPTIVE_LPT"):') == 1   # no duplication


def test_patch_migrates_from_hysteresis_to_congestion_aware():
    # A file already patched with the (post-hysteresis-fix, pre-congestion) block should also
    # upgrade in place, not just be recognized as "already newest".
    d = tempfile.mkdtemp()
    p = os.path.join(d, "scheduler.py")
    open(p, "w").write(FIXTURE.replace(H.ANCHOR, H.HYSTERESIS_BLOCK, 1))
    os.environ["VLLM_SCHED_PATH"] = p
    rc = H.main()
    out = open(p).read()
    assert rc == 0
    assert H.HYSTERESIS_BLOCK not in out
    assert H.CONGEST_BLOCK in out
    assert out.count('if os.getenv("ADAPTIVE_LPT"):') == 1


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
