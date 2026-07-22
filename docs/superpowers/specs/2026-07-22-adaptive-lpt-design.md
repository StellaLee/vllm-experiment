# Adaptive Per-Request Prefill-Length Threshold — Design

**Date:** 2026-07-22
**Status:** approved (design), pending implementation plan
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2

## Goal

Build a controller that makes `--long-prefill-token-threshold` **adaptive over time**, while the
step-wide budget (`--max-num-batched-tokens`) stays **static** at mono's value (16384). Test it
against the existing 8-arm lengthgate/threshold grid (see
`findings/2026-07-22-lengthgate-and-per-request-cap.md`) on the same phase-scheduled whale-fraction
workload, to see whether it recovers `16384lpt512`'s +5% Phase-S throughput tax while keeping its
Phase-W tail protection.

## Why this, and why now

This session established a chain of results that leads directly here:

1. **Step-wide budget controllers (hslo, lengthgate) both hit structural walls** — hslo has no
   load-sensitive signal to react to (decode cost is flat on this hardware); lengthgate's
   step-wide toggle collaterally starves bystanders while protecting (bug 2, unfixed).
2. **The native per-request cap (`long_prefill_token_threshold`) avoids both walls**, tested as a
   *static* value (`16384lpt512`): pareto-dominates static-2048 on both tail metrics
   (W:max −43%, W:goodput +1.9pp) for a throughput cost of +5% S:TTFT.
3. **That +5% cost is pure waste.** It comes from the threshold slicing ordinary non-whale prompts
   during Phase S, where whale-fraction is 0 and there's nothing to protect against. A controller
   that relaxes the threshold to "off" when no long prompt is around, and drops it to "protect"
   only when one is, should recover close to mono's throughput without giving up any of the tail
   protection — because unlike lengthgate, **the step budget itself never moves**, so there is no
   mechanism left for one request's protection to cost another request anything.
4. **The threshold×budget grid found the threshold's own value is non-monotonic** (512 beats 256)
   and that pairing it with a smaller step budget adds nothing (`2048lpt512` ≈ `16384lpt512`) —
   both results argue for keeping the budget static and touching only the threshold, and for
   reusing 512 (the one validated value) rather than guessing a new one.

## Mechanism

`long_prefill_token_threshold` is read as a static `scheduler_config` attribute at both existing
use sites (`scheduler.py:736` running-loop, `:1032` waiting-loop), with no caching — so setting it
once per scheduling step is enough for both sites to pick up the new value automatically. This is
simpler than lengthgate's hook, which needed four coordinated edits (signature, dispatch, method,
call site) because it plumbed a value through `ChunkSizeController`. This controller only needs
**one** insertion point, and it stays fully decoupled from `ChunkSizeController`/`DYNAMIC_CHUNK` —
the step budget genuinely never changes.

Insert immediately after `token_budget = self.max_num_scheduled_tokens` (line 675), which runs
unconditionally on every scheduling step, before the `if self._chunk_ctrl is not None:` block:

```python
if os.getenv("ADAPTIVE_LPT"):
    _alpt_pf = 0
    for _r in self.running:
        if getattr(_r, 'is_prefill_chunk', False):
            _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
    for _r in self.waiting:
        _alpt_pf = max(_alpt_pf, _r.num_prompt_tokens - _r.num_computed_tokens)
    gate = int(os.getenv("ADAPTIVE_LPT_GATE", "4096"))
    protect = int(os.getenv("ADAPTIVE_LPT_PROTECT", "512"))
    off = int(os.getenv("ADAPTIVE_LPT_OFF", "0"))
    self.scheduler_config.long_prefill_token_threshold = protect if _alpt_pf > gate else off
```

- `_alpt_pf` (max prefill-tokens-still-needed over running prefill-chunks + waiting requests) is
  the exact same signal lengthgate already validates — computed independently here (not reused
  from lengthgate's own computation, since that only runs when `_chunk_ctrl is not None`, which
  must not be a precondition for this controller).
- `gate=4096`, `protect=512` reuse lengthgate's already-validated values as-is (decided during
  brainstorming: this experiment's only new variable is *when* the cap engages, not *what* value it
  uses — keeps the test focused on one question).
- `off=0` uses vLLM's own semantics for "disabled" (`if 0 < threshold < num_new_tokens`), so the
  off state exactly reproduces mono's uncapped admission — the cleanest possible baseline to
  measure the recovered throughput against.
- Gated on `os.getenv("ADAPTIVE_LPT")` so one patched binary serves both the new arm (`ADAPTIVE_LPT=1`)
  and, if ever needed, an unmodified-behavior arm (unset) without re-patching.

## Components

1. **`scripts/hotpatch_adaptive_lpt.py`** — idempotent single-anchor patcher, following
   `hotpatch_lengthgate.py`'s defensive pattern: locate the anchor text, error if not found, check
   for a unique marker (e.g. `"ADAPTIVE_LPT"` string presence) before patching so re-application is
   a no-op.
2. **`tests/test_hotpatch_adaptive_lpt.py`** — fixture-based structural test (a small string
   mimicking scheduler.py's shape around line 675), verifying the block lands at the right spot and
   re-application is a no-op. No test needed for the gate arithmetic itself (a one-line ternary);
   it's exercised by the live run and the trace.
3. **`ADAPTIVE_LPT_TRACE=<path>`** — per-step CSV logger (`step,pf_remaining,threshold`), mirroring
   `DYNAMIC_CHUNK_TRACE`. **Mandatory to review before reporting any result** — lengthgate had two
   real bugs (depth==0 loophole, admission-starvation) that were only caught by exactly this kind
   of trace forensics, not by looking at the summary numbers.
4. **`run_adaptive_lpt_arm.sh`** — launches the new `adaptivelpt` arm (`--max-num-batched-tokens
   16384`, `ADAPTIVE_LPT=1 ADAPTIVE_LPT_GATE=4096 ADAPTIVE_LPT_PROTECT=512 ADAPTIVE_LPT_OFF=0`,
   `DYNAMIC_CHUNK=0`, `PREFIX_REORDER=0`) over the identical phase-schedule workload used by every
   other arm this session, then re-analyzes `ARMS` extended with `adaptivelpt` into
   `logs/lgate_ANALYSIS_v5.txt`.

## Setup (identical to every prior lengthgate/threshold arm, for comparability)

- Model Qwen2.5-Coder-14B-Instruct, TP=2, GPUs 0,1, `--gpu-memory-utilization 0.90`,
  `--max-model-len 16384`, `--max-num-seqs 128`.
- Workload: `--phase-schedule "6:0.0@60,1.0:0.2@60" --duration 360` (Phase S: rate=6/s, 0% whales,
  60s; Phase W: rate=1/s, 20% whales, 60s; 3 full cycles).
- Client: single-turn, `--max-tokens 256`, `--pad-mean-chars 800 --pad-cv2 0.5 --pad-min 100
  --pad-max 8000`, whales `--whale-min-chars 44000 --whale-max-chars 50000`,
  `--max-prompt-chars 50000`, `--pad-seed 1001` (whale positions paired across every arm).

## Success criteria (single trial, directional)

- **S:TTFT** close to mono's 250ms (the off-state should reproduce mono almost exactly whenever
  `pf_remaining ≤ 4096`).
- **W:max / W:goodput** at least as good as `16384lpt512`'s (≤490ms / ≥100%) — since protect mode
  uses the identical validated value, just gated in time instead of always-on.
- If both hold, this is the arm that recovers `16384lpt512`'s throughput tax without giving back
  any of its tail protection.
- Consistent with every other result in this line of work: report as n=1, directional; replicate
  before treating any specific number as final.

## Non-goals

- Not sweeping `gate=4096` or `protect=512` in this round — both reuse already-validated values;
  a follow-up sweep is a separate experiment.
- Not testing on the continuum-length (non-binary) workload — deferred; this test reuses the exact
  binary phase-schedule workload for direct comparability with the existing 8-arm table.
- No changes to `ChunkSizeController` or any `DYNAMIC_CHUNK`/`CHUNK_MODE` code path.
- No fairness/priority-based reordering — purely the length-gated per-request cap, same scope as
  lengthgate.
