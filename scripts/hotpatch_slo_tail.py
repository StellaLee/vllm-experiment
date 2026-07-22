#!/usr/bin/env python3
"""Add CHUNK_MODE=slotail: drive the chunk budget off a windowed high PERCENTILE of
per-step latency instead of its EMA/mean (CHUNK_MODE=slo). The mean signal is blind in the
sensitive regime -- frequent cheap decode-only steps dominate the average and wash out the
rare expensive prefill steps, so the controller never engages (pins at ceiling). Keying off
the tail (p99 by default) makes a prefill stall visible, so the controller can react.

Replaces the whole ChunkSizeController class, keeping depth + slo (mean) + slotail (tail)
so all three can be compared in one experiment. Run patch_scheduler.py FIRST. Re-runnable.

New env (slotail only):
  CHUNK_MODE=slotail
  DYNAMIC_CHUNK_PCTL     percentile of the step-latency window (default 99)
  DYNAMIC_CHUNK_WINDOW   window size in steps (default 128)
  DYNAMIC_CHUNK_WINMIN   min samples before acting (default 20)
  CHUNK_MODE=slocvar       tail-MEAN (CVaR) variant of slotail (steadier than a p99 point)
  DYNAMIC_CHUNK_CVAR_PCTL  slocvar tail cutoff; signal = mean of worst (100-p)% (default 90)
  DYNAMIC_CHUNK_TRACE      path to a per-step csv (step,wall_s,depth,signal_ms,chunk) so the
                           budget trajectory over time is recorded; unset = no trace (all modes)
  DYNAMIC_CHUNK_START      initial chunk budget (default = min/floor). Start at the deployed
                           chunked-prefill point and grow up, NOT at the mono ceiling (all modes)
  (reuses DYNAMIC_CHUNK_SLO_MS, DYNAMIC_CHUNK_STEP, DYNAMIC_CHUNK_MIN, DYNAMIC_CHUNK_WINDOW)
"""
import os, re, sys
from pathlib import Path


def _find_sched() -> Path:
    """Locate scheduler.py in the *active* vLLM install (env override wins)."""
    override = os.environ.get("VLLM_SCHED_PATH")
    if override:
        return Path(override)
    import vllm  # noqa: PLC0415 -- resolved against the running interpreter
    return Path(vllm.__file__).parent / "v1" / "core" / "sched" / "scheduler.py"


SCHED = _find_sched()
if not SCHED.exists():
    print(f"ERROR: {SCHED} not found", file=sys.stderr); sys.exit(1)
src = SCHED.read_text()

if "last_tokens=None" in src:
    print("feedforward v2 (actual-token) controller already present — no changes."); sys.exit(0)

NEW_CLASS = '''class ChunkSizeController:
    """Chunk-prefill token-budget controller. Modes: depth | slo (EMA/mean) |
    slotail (windowed p-percentile) | slocvar (windowed tail-mean / CVaR).
    Enable with DYNAMIC_CHUNK=1, pick via CHUNK_MODE."""

    def __init__(self, min_tokens: int, max_tokens: int, target: int,
                 hold: int = 3) -> None:
        import os
        from collections import deque
        self.min = min_tokens
        self.max = max_tokens
        # Start at the DEPLOYED chunked-prefill operating point, not the ceiling:
        # DYNAMIC_CHUNK_START (default = min/floor = the small-chunk strategy vLLM runs),
        # then AIMD-grow toward max only when the latency tail has headroom. Starting at
        # max (mono) begins in the worst-TPOT regime and biases the controller upward.
        _start = int(os.getenv("DYNAMIC_CHUNK_START", str(min_tokens)))
        self.chunk = max(min_tokens, min(max_tokens, _start))
        self.target = target
        self.hold = hold
        self._step_count = 0
        self._shrink_count = 0
        self._grow_count = 0
        self.mode = os.getenv("CHUNK_MODE", "depth").strip().lower()
        self.slo = float(os.getenv("DYNAMIC_CHUNK_SLO_MS", "50")) / 1000.0
        self.aimd_step = int(os.getenv("DYNAMIC_CHUNK_STEP", str(max(1, min_tokens))))
        self.ema = float(os.getenv("DYNAMIC_CHUNK_EMA", "0.3"))
        self._last_t = None
        self._lat = None
        # slotail config
        self._pctl = float(os.getenv("DYNAMIC_CHUNK_PCTL", "99"))
        self._win = deque(maxlen=int(os.getenv("DYNAMIC_CHUNK_WINDOW", "128")))
        self._win_min = int(os.getenv("DYNAMIC_CHUNK_WINMIN", "20"))
        # slocvar: CVaR/expected-shortfall cutoff. signal = MEAN of the worst
        # (100 - cvar_pctl)% of the window -> lower variance than a single percentile.
        self._cvar_pctl = float(os.getenv("DYNAMIC_CHUNK_CVAR_PCTL", "90"))
        # feedforward: per-token step-time estimate (EMA of dt / granted budget) and the
        # budget granted last step (needed to attribute the measured latency to a token count).
        self._tau = None
        self._last_budget = None
        # Per-step budget trace (DYNAMIC_CHUNK_TRACE=path). Line-buffered so a
        # SIGTERM'd server still leaves a complete csv; one short write / ~30ms
        # decode step is negligible vs the step itself.
        self._trace_fh = None
        _tp = os.getenv("DYNAMIC_CHUNK_TRACE", "").strip()
        if _tp:
            try:
                self._trace_fh = open(_tp, "w", buffering=1)
                print("step,wall_s,depth,signal_ms,chunk", file=self._trace_fh)
            except Exception:
                self._trace_fh = None
        import logging
        logging.getLogger(__name__).info(
            "ChunkSizeController mode=%s min=%d max=%d slo_ms=%.1f pctl=%.0f",
            self.mode, self.min, self.max, self.slo * 1000.0, self._pctl)

    def step(self, decode_depth: int, last_tokens=None) -> int:
        self._step_count += 1
        if self.mode == "feedforward":
            return self._step_feedforward(decode_depth, last_tokens)
        if self.mode == "slocvar":
            return self._step_slocvar(decode_depth)
        if self.mode == "slotail":
            return self._step_slotail(decode_depth)
        if self.mode == "slo":
            return self._step_slo(decode_depth)
        return self._step_depth(decode_depth)

    def _trace(self, decode_depth: int, signal_ms: float) -> None:
        if self._trace_fh is not None:
            import time
            print("%d,%.3f,%d,%.2f,%d" % (
                self._step_count, time.time(), decode_depth, signal_ms, self.chunk),
                file=self._trace_fh)

    def _step_depth(self, decode_depth: int) -> int:
        if decode_depth > self.target * 1.5:
            self._shrink_count += 1; self._grow_count = 0
        elif decode_depth < self.target * 0.5:
            self._grow_count += 1; self._shrink_count = 0
        else:
            self._shrink_count = 0; self._grow_count = 0
        if self._shrink_count >= self.hold:
            self.chunk = max(self.min, self.chunk // 2); self._shrink_count = 0
        elif self._grow_count >= self.hold:
            self.chunk = min(self.max, self.chunk * 2); self._grow_count = 0
        self._trace(decode_depth, float(decode_depth))
        return self.chunk

    def _measure(self, decode_depth):
        import time
        now = time.monotonic(); dt = None
        if self._last_t is not None:
            d = now - self._last_t
            if decode_depth > 0 and d < 0.5:
                dt = d
        self._last_t = now
        return dt

    def _step_slo(self, decode_depth: int) -> int:
        dt = self._measure(decode_depth)
        if dt is not None:
            self._lat = dt if self._lat is None else (
                self.ema * dt + (1.0 - self.ema) * self._lat)
        if self._lat is not None:
            if self._lat > self.slo:
                self.chunk = max(self.min, self.chunk // 2)
            elif self._lat < self.slo * 0.75:
                self.chunk = min(self.max, self.chunk + self.aimd_step)
        self._trace(decode_depth, (self._lat or 0.0) * 1000.0)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[slo] step=%d depth=%d ema_ms=%.1f chunk=%d",
                self._step_count, decode_depth, (self._lat or 0.0) * 1000.0, self.chunk)
        return self.chunk

    def _step_slotail(self, decode_depth: int) -> int:
        dt = self._measure(decode_depth)
        if dt is not None:
            self._win.append(dt)
        sig = 0.0
        if len(self._win) >= self._win_min:
            xs = sorted(self._win)
            k = min(len(xs) - 1, int(self._pctl / 100.0 * len(xs)))
            sig = xs[k]  # windowed tail-percentile step latency
            if sig > self.slo:
                self.chunk = max(self.min, self.chunk // 2)
            elif sig < self.slo * 0.75:
                self.chunk = min(self.max, self.chunk + self.aimd_step)
        self._trace(decode_depth, sig * 1000.0)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[slotail] step=%d depth=%d p%.0f_ms=%.1f chunk=%d",
                self._step_count, decode_depth, self._pctl, sig * 1000.0, self.chunk)
        return self.chunk

    def _step_feedforward(self, decode_depth: int, last_tokens=None) -> int:
        # Model-based (feedforward) sizing: pick the token budget whose PREDICTED step latency
        # hits the SLO, instead of servoing measured latency to a setpoint (which limit-cycles
        # on the bimodal plant -> min/max only). tau = EMA of per-token step time (dt / tokens
        # ACTUALLY processed last step). Budget B* = SLO / tau is the self-consistent point where
        # a full step takes exactly the SLO -> converges to a STABLE INTERIOR value. Decode takes
        # its share of B*, prefill gets the rest, so as the decode batch grows the prefill chunk
        # auto-shrinks. v2: divide by ACTUAL tokens (passed from the scheduler), not the granted
        # budget -- on decode-only steps granted>>actual, which made v1 underestimate tau and
        # inflate the budget to the ceiling. Falls back to granted budget if not supplied.
        import time
        now = time.monotonic()
        T = last_tokens if (last_tokens is not None and last_tokens > 0) else self._last_budget
        if self._last_t is not None and T:
            dt = now - self._last_t
            if 0.0 < dt < 0.5:  # ignore idle gaps between bursts
                tau_obs = dt / float(max(1, T))
                self._tau = tau_obs if self._tau is None else (
                    self.ema * tau_obs + (1.0 - self.ema) * self._tau)
        self._last_t = now
        if self._tau and self._tau > 0.0:
            self.chunk = int(max(self.min, min(self.max, round(self.slo / self._tau))))
        self._last_budget = self.chunk
        self._trace(decode_depth, (self._tau or 0.0) * self.chunk * 1000.0)  # predicted step ms
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[feedforward] step=%d depth=%d tau_us=%.3f pred_ms=%.1f chunk=%d",
                self._step_count, decode_depth, (self._tau or 0.0) * 1e6,
                (self._tau or 0.0) * self.chunk * 1000.0, self.chunk)
        return self.chunk

    def _step_slocvar(self, decode_depth: int) -> int:
        # CVaR / expected-shortfall control: signal = MEAN of the window tail at/above
        # cvar_pctl (default = worst 10%). Averaging the tail is a far lower-variance
        # estimator than slotail's single p99 order statistic -> steadier control.
        dt = self._measure(decode_depth)
        if dt is not None:
            self._win.append(dt)
        sig = 0.0
        if len(self._win) >= self._win_min:
            xs = sorted(self._win)
            k = min(len(xs) - 1, int(self._cvar_pctl / 100.0 * len(xs)))
            tail = xs[k:]
            sig = sum(tail) / len(tail)  # windowed tail-mean (CVaR)
            if sig > self.slo:
                self.chunk = max(self.min, self.chunk // 2)
            elif sig < self.slo * 0.75:
                self.chunk = min(self.max, self.chunk + self.aimd_step)
        self._trace(decode_depth, sig * 1000.0)
        if self._step_count % 50 == 0:
            import logging
            logging.getLogger(__name__).info(
                "ChunkCtrl[slocvar] step=%d depth=%d cvar%.0f_ms=%.1f chunk=%d",
                self._step_count, decode_depth, self._cvar_pctl, sig * 1000.0, self.chunk)
        return self.chunk

'''

pattern = re.compile(
    r"class ChunkSizeController:.*?\n\n(?=class Scheduler\(SchedulerInterface\):)",
    re.DOTALL)
if not pattern.search(src):
    print("ERROR: ChunkSizeController class not found. Run patch_scheduler.py first.",
          file=sys.stderr); sys.exit(1)
src = pattern.sub(NEW_CLASS, src, count=1)
SCHED.write_text(src)
print(f"slotail+slocvar controller installed in {SCHED}")
print("Enable slotail: DYNAMIC_CHUNK=1 CHUNK_MODE=slotail DYNAMIC_CHUNK_MIN=512 "
      "DYNAMIC_CHUNK_SLO_MS=50 DYNAMIC_CHUNK_PCTL=99")
print("Enable slocvar: DYNAMIC_CHUNK=1 CHUNK_MODE=slocvar DYNAMIC_CHUNK_MIN=512 "
      "DYNAMIC_CHUNK_SLO_MS=50 DYNAMIC_CHUNK_CVAR_PCTL=90")
