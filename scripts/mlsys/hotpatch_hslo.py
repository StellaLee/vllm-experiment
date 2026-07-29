#!/usr/bin/env python3
"""Add the SLO-headroom feedforward controller (CHUNK_MODE=hslo) to vLLM's ChunkSizeController.

Idempotent. Two edits to scheduler.py:
  1. dispatch: route mode=="hslo" to _step_hslo in ChunkSizeController.step()
  2. method:   insert _step_hslo just before `class Scheduler(SchedulerInterface):`

Design (docs/superpowers/specs/2026-07-22-slo-headroom-feedforward-design.md):
  iter_time ~= decode_baseline + alpha * prefill_tokens
  budget    = clamp((SLO - decode_baseline)/alpha, floor, ceiling)
Estimator is SEGREGATED so the two costs never blend: decode_baseline = EMA(dt) on pure-decode
steps (prefill_last==0); alpha = EMA((dt-decode_baseline)/prefill_last) on prefill-heavy steps
(prefill_last >= ALPHA_MIN_PREFILL). A single EMA(dt/total_tokens) is dominated by cheap decode
steps and overestimates alpha ~4x -> rails to floor (the slocvar/ffv2/depth failure).
"""
import os
import sys
from pathlib import Path


def _find_sched() -> Path:
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


DISPATCH_ANCHOR = (
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)
DISPATCH_NEW = (
    '        if self.mode == "hslo":\n'
    '            return self._step_hslo(decode_depth, last_tokens)\n'
    '        if self.mode == "slo":\n'
    '            return self._step_slo(decode_depth)\n'
)

METHOD_ANCHOR = "\nclass Scheduler(SchedulerInterface):\n"
METHOD = '''
    def _step_hslo(self, decode_depth: int, last_tokens=None) -> int:
        # SLO-headroom FEEDFORWARD. Size the prefill budget from MEASURED decode cost plus a
        # feedforward prefill term, so a whale is capped on the iteration it ARRIVES -- not one
        # step late like the reactive controllers (a whale prefills in a single iteration, so any
        # controller reading the PREVIOUS step's latency is structurally too slow).
        #   decode_baseline (_db, seconds) = EMA(dt) over PURE-DECODE steps (prefill_last == 0)
        #   alpha (_alpha, sec/prefill-token) = EMA((dt - _db)/prefill_last) over PREFILL-HEAVY
        #       steps (prefill_last >= _alpha_min) only
        # The two costs are never blended: a lone EMA(dt/total_tokens) is dominated by the many
        # cheap decode steps -> overestimates alpha ~4x -> budget rails to the floor.
        #   budget = clamp((SLO - _db)/alpha, floor, ceiling)
        # prefill_last is derived, no new plumbing: last_tokens (actual processed last step) minus
        # last step's decode depth (each decoder advances exactly 1 token/iteration).
        import time
        if not getattr(self, "_hslo_init", False):
            self._hslo_start = self.chunk          # warmup / no-estimate budget
            self._db = None                        # decode_baseline, seconds
            self._alpha = None                     # seconds per prefill token
            self._hslo_prev_depth = 0
            self._alpha_min = int(os.getenv("DYNAMIC_CHUNK_ALPHA_MIN_PREFILL", "256"))
            # alpha hardware floor (offline-profiled): clamps the ONLINE alpha estimate from below so
            # a noisy near-zero reading can't rail the budget to the ceiling. With alpha >= alpha_hw,
            # budget = (SLO-db)/alpha_eff <= (SLO-db)/alpha_hw, so predicted_iter = db+budget*alpha_eff
            # <= SLO BY CONSTRUCTION -> the SLO becomes a hard TBT guarantee, not just a target.
            # Env DYNAMIC_CHUNK_ALPHA_MIN is ms/token; 0 (default) disables the floor.
            self._alpha_hw = float(os.getenv("DYNAMIC_CHUNK_ALPHA_MIN", "0.0")) / 1000.0
            self._hslo_init = True
        now = time.monotonic()
        if self._last_t is not None and last_tokens is not None:
            dt = now - self._last_t
            prefill_last = max(0, int(last_tokens) - self._hslo_prev_depth)
            if prefill_last == 0 and self._hslo_prev_depth > 0 and 0.0 < dt < 0.5:
                # real decode-only iteration -> calibrate decode_baseline
                self._db = dt if self._db is None else (self.ema * dt + (1.0 - self.ema) * self._db)
            elif prefill_last >= self._alpha_min and 0.0 < dt < 8.0:
                # prefill-heavy iteration (incl. the 6s whale, which feedforward's <0.5s guard
                # discarded) -> back out per-prefill-token cost from the residual over decode
                a = max(dt - (self._db or 0.0), 1e-4) / float(prefill_last)
                self._alpha = a if self._alpha is None else (self.ema * a + (1.0 - self.ema) * self._alpha)
        self._last_t = now
        self._hslo_prev_depth = decode_depth

        if decode_depth == 0:
            self.chunk = self.max                  # nothing to protect -> full-speed whale TTFT
        elif self._db is None or self._alpha is None or self._alpha <= 0.0:
            self.chunk = self._hslo_start          # warmup, costs not yet learned
        else:
            headroom = self.slo - self._db
            if headroom <= 0.0:
                self.chunk = self.min              # decode alone already blows SLO; minimize prefill
            else:
                self.chunk = int(max(self.min, min(self.max, headroom / max(self._alpha, self._alpha_hw))))
        self._trace(decode_depth, (self._db or 0.0) * 1000.0)   # signal_ms = decode_baseline
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[hslo] step=%d depth=%d db_ms=%.1f alpha_us=%.3f head_ms=%.1f chunk=%d",
                self._step_count, decode_depth, (self._db or 0.0) * 1000.0,
                (self._alpha or 0.0) * 1e6, (self.slo - (self._db or 0.0)) * 1000.0, self.chunk)
        return self.chunk

'''


def main() -> int:
    sched = _find_sched()
    src = sched.read_text()
    if "def _step_hslo" in src:
        print("hslo controller already present -- no changes.")
        return 0
    if DISPATCH_ANCHOR not in src:
        print("ERROR: step() dispatch anchor (mode=='slo') not found", file=sys.stderr)
        return 1
    if METHOD_ANCHOR not in src:
        print("ERROR: 'class Scheduler(SchedulerInterface):' anchor not found", file=sys.stderr)
        return 1
    src = src.replace(DISPATCH_ANCHOR, DISPATCH_NEW, 1)
    src = src.replace(METHOD_ANCHOR, METHOD + METHOD_ANCHOR, 1)
    sched.write_text(src)
    print(f"hslo controller installed in {sched}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
