# Congestion-Aware Adaptive Prefill-Length Threshold — Design

**Date:** 2026-08-27
**Status:** approved (design), implemented, calibration in progress
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2

**Revision (2026-08-27, same day):** switched the congestion gate from an absolute running-count
threshold to a **fraction of `max_num_running_reqs`** (`ADAPTIVE_LPT_CONGEST_FRAC`/
`_EXIT_FRAC`, replacing the originally-drafted `_CONGEST_GATE`/`_EXIT_GATE`). Reasoning: an
absolute count is a magic number tied to this box's `--max-num-seqs 128` and would need
recalibrating on any other configuration — a knob that's overfit to one deployment is exactly the
"too many knobs tuned to fit the traffic" failure mode this whole controller line has been trying
to avoid. A fraction of configured concurrency is self-normalizing and doesn't need
recalibration if `--max-num-seqs` changes, even though the fraction's cutoff value itself still
needs one calibration pass. The rest of this document (mechanism narrative, motivation) is
unchanged; code blocks below reflect the fraction-based version actually implemented in
`scripts/mlsys/hotpatch_adaptive_lpt.py`.

## Goal

Extend the existing `adaptivelpt2` controller (`scripts/hotpatch_adaptive_lpt.py`, hysteresis
version) with a second, independent gate on **current system congestion**, so the per-request cap
degrades toward step-wide throttling exactly when — and only when — extending a long request's
residency would otherwise be expensive. Test across the same three rate points already run
(WRATE=0.5, 1.0, 2.0) to check whether this closes the WRATE=2.0 collapse without regressing the
WRATE=1.0 win or the WRATE=0.5 behavior.

## Why this, and why now

This session ran the previously-untested rate sweep for `adaptivelpt2` and found the "wins both
axes" result (`findings/2026-07-22-lengthgate-and-per-request-cap.md`) holds only at the exact
calibrated rate (WRATE=1.0) it was demonstrated at:

- **WRATE=0.5**: adaptive doesn't tie mono on S:TTFT (284 vs 268ms) and is dominated by the
  simpler always-on `16384lpt512` arm on every axis measured.
- **WRATE=2.0**: catastrophic collapse — S:TTFT more than doubles vs mono (18994 vs 8405ms) and
  W:max is *worse* than mono's own already-bad tail (16351.5 vs 6241.9ms). `static-512` wins
  outright on both axes at this point (S:TTFT 7574 < mono's 8405; W:max 143.1, goodput 100%).

Root cause, confirmed directly from the WRATE=2.0 chunktrace: one protect episode (a single whale's
full prefill under the cap) lasted **226.6 seconds** — longer than the 60s Phase-W window it
started in, so subsequent whales piled up behind it before it finished. Mechanism: the per-request
cap only throttles the *flagged* request to 512 tok/round; it leaves the step-wide budget
(`--max-num-batched-tokens 16384`) open for everything else. Under light/calibrated load there's
little else to fill that budget with, so each of the whale's ~27 rounds is fast. Under heavy load,
the backlog of concurrent decode work fills that budget every round, so each round gets slow,
multiplying the whale's total residency — which itself deepens the backlog behind it. `static-512`
avoids this because it bounds the *entire* step's work, not just the flagged request's slice, so
round duration stays bounded regardless of backlog.

The controller as built has no signal for this: it only asks "is a long prefill present?"
(`pf_remaining`), never "is the system currently congested enough that protecting this request the
usual way will backfire?" This design adds that second signal.

## Mechanism

Reuse the exact `pf_remaining` gate and hysteresis from `adaptivelpt2` unchanged (entry
`ADAPTIVE_LPT_GATE=4096`, exit `ADAPTIVE_LPT_EXIT_GATE=0`, protect value
`ADAPTIVE_LPT_PROTECT=512`) — that part is validated and stays as-is. Add a second, independently
gated signal:

- **Congestion proxy**: `len(self.running) / self.max_num_running_reqs` — current load as a
  **fraction** of configured concurrency (`--max-num-seqs`), not an absolute sequence count.
  `max_num_running_reqs` is set from `scheduler_config.max_num_seqs` and already used elsewhere in
  the real scheduler as the running-list's own capacity bound (`assert len(self.running) <=
  self.max_num_running_reqs`), so this is a natural, already-available normalizer — the gate
  doesn't need recalibrating if `--max-num-seqs` changes on a different deployment.
- **Second hysteresis gate**, same asymmetric-entry/exit pattern as the `pf_remaining` fix (a
  shared bar was exactly what caused the earlier hysteresis bug — flapping/premature release):
  `ADAPTIVE_LPT_CONGEST_FRAC` (entry, load fraction in `(0, 1]`) and
  `ADAPTIVE_LPT_CONGEST_EXIT_FRAC` (exit, must be `<=` entry). Tracked as separate per-step state
  (`self._alpt_in_congest`), independent of `self._alpt_in_protect`.
- **Degraded action**: when both `_alpt_in_protect` and `_alpt_in_congest` are true, shrink the
  step-wide `token_budget` itself (not just the flagged request's slice) to
  `ADAPTIVE_LPT_CONGEST_BUDGET` (default 512 — reusing the one step-wide value already proven to
  win outright at WRATE=2.0, i.e. composing the two already-validated arms rather than inventing a
  new number). When not congested, behavior is byte-for-byte identical to today's `adaptivelpt2`
  (`token_budget` untouched, only the per-request threshold toggles).

```python
if os.getenv("ADAPTIVE_LPT"):
    _alpt_pf = 0
    for _r in self.running:
        if getattr(_r, 'is_prefill_chunk', False):
            _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
    for _r in self.waiting:
        _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
    _alpt_gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
    _alpt_exit_gate = int(os.getenv("ADAPTIVE_LPT_EXIT_GATE", "0"))
    _alpt_protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
    _alpt_off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
    _alpt_was_protect = getattr(self, "_alpt_in_protect", False)
    _alpt_enter = _alpt_pf > _alpt_gate
    _alpt_stay = _alpt_was_protect and _alpt_pf > _alpt_exit_gate
    _alpt_in_protect = _alpt_enter or _alpt_stay
    self._alpt_in_protect = _alpt_in_protect
    _alpt_thr = _alpt_protect if _alpt_in_protect else _alpt_off
    self.scheduler_config.long_prefill_token_threshold = _alpt_thr

    _alpt_congest_frac = float(os.getenv("ADAPTIVE_LPT_CONGEST_FRAC", "0"))  # 0 = feature off
    _alpt_running = len(self.running)
    _alpt_cap = max(1, self.max_num_running_reqs)
    _alpt_load = _alpt_running / _alpt_cap
    _alpt_in_congest = False
    if _alpt_congest_frac > 0:
        _alpt_congest_exit_frac = float(os.getenv("ADAPTIVE_LPT_CONGEST_EXIT_FRAC", str(_alpt_congest_frac)))
        _alpt_was_congest = getattr(self, "_alpt_in_congest", False)
        _alpt_c_enter = _alpt_load > _alpt_congest_frac
        _alpt_c_stay = _alpt_was_congest and _alpt_load > _alpt_congest_exit_frac
        _alpt_in_congest = _alpt_c_enter or _alpt_c_stay
        self._alpt_in_congest = _alpt_in_congest
        if _alpt_in_protect and _alpt_in_congest:
            _alpt_congest_budget = int(os.getenv("ADAPTIVE_LPT_CONGEST_BUDGET", "512"))
            token_budget = min(token_budget, _alpt_congest_budget)

    _alpt_trace = os.getenv("ADAPTIVE_LPT_TRACE")
    if _alpt_trace:
        import time as _alpt_time
        with open(_alpt_trace, "a") as _alpt_f:
            _alpt_f.write(
                f"{_alpt_time.monotonic()},{_alpt_pf},{_alpt_thr},"
                f"{_alpt_running},{_alpt_load:.4f},{int(_alpt_in_congest)},{token_budget}\n")
```

- Feature-flagged by `ADAPTIVE_LPT_CONGEST_FRAC=0` (default, off) so the patcher can be applied and
  tested for pure regression (identical to `adaptivelpt2`) before the congestion axis is enabled at
  all — isolates "did the refactor change anything" from "does the new gate help."
- `token_budget = min(token_budget, ...)` rather than a flat assignment, so this composes correctly
  if `_chunk_ctrl` (a different, unrelated controller) also mutates `token_budget` later in the same
  step — this block runs *before* that `if self._chunk_ctrl is not None:` check, same anchor point
  as today.
- Trace now logs `running` and `in_congest` per step — required to calibrate `CONGEST_FRAC` (see
  Non-goals) and to verify the hysteresis doesn't flap, the same trace-forensics discipline that
  caught both `adaptivelpt`/`lengthgate`'s earlier bugs.

## Components

1. **`scripts/hotpatch_adaptive_lpt.py`** — upgrade in place. Follows the file's own established
   migration pattern (it already detects and upgrades an older pre-hysteresis block); add a third
   `NEWER_BLOCK` and a migration path from the current hysteresis block to it, so an
   already-patched box can be upgraded without restoring from backup.
2. **`tests/test_hotpatch_adaptive_lpt.py`** — extend with: (a) a regression test that with
   `ADAPTIVE_LPT_CONGEST_FRAC` unset/0, the emitted block behaves identically to today's (structural
   check only, not a live-server test); (b) a structural test that the congestion block appears
   after the protect-mode assignment and before the trace-write line.
3. **`ADAPTIVE_LPT_TRACE`** — extended CSV schema (`wall_s,pf_remaining,threshold,running,
   in_congest,token_budget`), backward-incompatible with old trace files (fine — trace is
   per-run diagnostic output, not a stored artifact anything else parses positionally except ad hoc
   analysis scripts written per-run).
4. **Calibration pass** (not a script — an analysis step): before picking `CONGEST_FRAC`/
   `CONGEST_EXIT_FRAC` values, run one instrumented pass at WRATE=2.0 with the congestion axis
   *disabled* (`CONGEST_FRAC=0`) purely to log `running` per step and find where it sits during the
   known-collapsed regime vs. during WRATE=0.5/1.0's known-fine regime. Do not guess these numbers
   the way the original `SCHED` rate needed 4 bisection rounds — use the trace.
5. **`orchestrate/mlsys/rate_sweep_lpt_congest.sh`** — new arm `adaptivelptcongest`, parameterized
   on `WRATE`/`TAG`/`GPUS`/`PORT` like `rate_sweep_lpt_v2.sh`, adding
   `ADAPTIVE_LPT_CONGEST_FRAC`/`ADAPTIVE_LPT_CONGEST_EXIT_FRAC`/`ADAPTIVE_LPT_CONGEST_BUDGET` to the
   existing `run_adaptive()` env block. Reruns the same 4-arm comparison at each of the 3 already-
   established rate points (0.5, 1.0, 2.0), plus the new arm as a 5th.

## Setup (identical to the existing rate-sweep arms, for direct comparability)

- Model Qwen2.5-Coder-14B-Instruct, TP=2, `--gpu-memory-utilization 0.90`, `--max-model-len 16384`,
  `--max-num-seqs 128`.
- Workload: `--phase-schedule "6:0.0@60,${WRATE}:0.2@60" --duration 360`, `--max-tokens 256`,
  `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100 --pad-max 8000`, whales
  `--whale-min-chars 44000 --whale-max-chars 50000`, `--max-prompt-chars 50000`,
  `--pad-seed 1001` — same as `rate_sweep_lpt.sh`, so results are directly comparable to the
  0.5/1.0/2.0 numbers already collected this session.

## Success criteria (single trial per rate point, directional — replicate before treating as final)

- **WRATE=2.0**: `adaptivelptcongest` closes most of the gap to `static-512`'s S:TTFT (7574ms) and
  W:max (397.4ms/100% goodput) — does not need to beat it, just stop losing to mono.
- **WRATE=1.0**: retains `adaptivelpt2`'s tie-and-match (S:TTFT ≈248ms, W:max ≈489ms) — the
  congestion gate should never fire at this rate if calibrated correctly (this is itself a
  calibration check, not just an outcome).
- **WRATE=0.5**: at least matches today's `adaptivelptw05` (not required to fix the
  dominated-by-`lpt512` issue at this rate — that's a separate, lower-priority gap; light load was
  never the failure mode this design targets).
- If WRATE=2.0 improves substantially without regressing 1.0/0.5, this becomes the arm that
  dominates across the full tested load range — the "winning strategy" the paper needs. If it
  doesn't converge within roughly a week of calibration/iteration, abandon and fall back to framing
  the plain `adaptivelpt2` result as a bounded, honestly-scoped case study (see mock review,
  `docs/2026-08-27-mock-mlsys-review.md`, Pivot A).

## Non-goals

- Not tuning `ADAPTIVE_LPT_GATE`/`PROTECT`/`EXIT_GATE` (the `pf_remaining` axis) — reuse validated
  values unchanged; this design touches only the new congestion axis.
- Not trying alternate congestion proxies (decode-token volume, actual measured step wall-time) in
  this round — `len(self.running)` first because it's the cheapest and most directly tied to the
  trace evidence; a follow-up can compare proxies if this one doesn't calibrate cleanly.
- Not admission control (rejecting/delaying whale admission itself) — stays within chunk-size
  control, consistent with the paper's positioning against LPRS/APC (which do admission-side
  control) in `paper-mlsys/tex/sections/D-related-work.tex`.
- No changes to `ChunkSizeController`/`DYNAMIC_CHUNK`/`CHUNK_MODE` code paths.
- Not testing the widened whale-size distribution (`wdist`) in this round — isolate the congestion
  fix to the rate axis first, using the exact workload already characterized at all 3 rate points.
