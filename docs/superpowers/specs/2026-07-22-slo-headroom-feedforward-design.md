# SLO-Headroom Feedforward Chunk Controller — Design

**Date:** 2026-07-22
**Mode name:** `CHUNK_MODE=hslo` (headroom-SLO)
**Context:** Fifth dynamic chunk controller. The prior four (slocvar, ffv1, ffv2, depth) all
degenerate — each rails to ceiling (≈mono) or floor (≈static-512) and none finds the interior
2048 knee. This controller is the synthesis: size the prefill budget from *measured decode cost*
(not a count, not last step's mixed latency) plus a *feedforward* prefill term known before the
whale is scheduled.

## Goal

Per scheduler step, pick `token_budget` so that the predicted next-iteration time stays under an
SLO, protecting in-flight decoders from whale-prefill freezes **without** over-taxing TTFT when
there is no decode work to protect. Target: match chunk-2048's TBT-p99/max win while keeping
TTFT closer to mono than static-512 does — i.e. actually land in the interior.

## Core model

An iteration's wall time decomposes as:

```
iter_time ≈ decode_baseline + α · prefill_tokens
```

- `decode_baseline` — whole-iteration decode cost (all active decoders, 1 token each). This is
  the **cost** term the user's critique demanded: a shallow-but-slow decode batch shows up here
  as a high baseline, correctly shrinking available headroom. Depth alone could not see this.
- `α` — marginal cost per prefill token.
- `prefill_tokens` — tokens we grant to prefill this step = the budget we are choosing. This term
  is **feedforward**: we know the chunk size before we schedule, so the whale is caught in time,
  unlike the reactive controllers that learn cost only after the 6 s iteration.

Budget solves `decode_baseline + α · budget ≤ SLO`:

```
budget = clamp( (SLO − decode_baseline) / α, floor, ceiling )
```

## Estimator — CORRECTION to the earlier (i) recommendation

Earlier I recommended getting `decode_baseline` via **(i)**: `EMA(dt − α̂·prefill)` every step,
with `α̂ = EMA(dt / total_tokens)`, claiming the EMAs self-correct. **Writing it out, that α̂ is
biased and would rail to the floor** — reproducing the exact failure we're trying to escape:

- A decode-only step has `dt≈28 ms`, `total_tokens≈20` → `α̂ ≈ 1.4 ms/tok`.
- A whale prefill step has `dt≈6000 ms`, `total_tokens≈16384` → `α̂ ≈ 0.37 ms/tok`.

`EMA(dt/total_tokens)` is dominated by the many cheap decode steps, so it **overestimates**
per-prefill-token cost several-fold → `headroom/α̂` collapses → budget pins to the floor. That is
slocvar/ffv2/depth all over again.

**Corrected estimator (segregated, closer to option (ii)):** never mix the two costs in one
average. Update each from the steps that isolate it.

- **Decode-dominated step** (`prefill_last == 0`, pure decode): `decode_baseline = EMA(dt)`.
  This is the direct decode-only measurement — cleaner signal, and under our whale workload
  pure-decode steps are common (whales are ~15% of requests, most iterations are decode-only).
- **Prefill-heavy step** (`prefill_last ≥ ALPHA_MIN_PREFILL`, default 256): back out α from the
  residual after removing known decode cost:
  `α = EMA( max(dt − decode_baseline, ε) / prefill_last )`.
- **Mixed small-prefill steps** (0 < prefill_last < ALPHA_MIN_PREFILL): update neither; too noisy
  to attribute.

`prefill_last` is derived, not newly plumbed: total tokens processed last step minus last step's
decode depth (each decoder contributes exactly 1 token in continuous batching):
`prefill_last = max(0, last_tokens − prev_decode_depth)`.

EMA weight `λ = 0.3` for both.

## Zero-decode exception (legitimate count check)

If `decode_depth == 0` there are no decoders to protect → return `ceiling`. This is a
presence check (is there decode work at all?), not a cost proxy, so it does not reintroduce the
count-not-cost flaw. Effect: an idle server prefills a whale at full budget → preserves whale
TTFT exactly when nothing is at risk.

## Warmup / degenerate guards

- Until `decode_baseline` and `α` have each had ≥1 update: return `start` (= ceiling).
- `headroom = SLO_MS − decode_baseline ≤ 0` (decode alone already blows SLO): return `floor` —
  can't help TBT by adding prefill; minimize it. (Expected rare; flagged in trace.)
- Always clamp to `[floor, ceiling]` and return `int`.

## Interface

Threads into the existing `ChunkSizeController` (scheduler.py ~line 67), dispatched from
`step(decode_depth, last_tokens)` (~line 122); hook already fires at ~line 594 with both args,
and `_ff_last_tokens` is already wired at ~line 1128. New state only: `_last_t` (monotonic),
`_prev_decode_depth`, `decode_baseline`, `alpha`. Reuses `_trace`.

**Env knobs:**

| env | meaning | default |
|---|---|---|
| `CHUNK_MODE=hslo` | select this controller | — |
| `DYNAMIC_CHUNK_SLO_MS` | per-iteration time target (ms) | 50 |
| `DYNAMIC_CHUNK_MIN` | floor | 512 |
| `DYNAMIC_CHUNK_START` | ceiling / start / warmup budget | 16384 |
| `DYNAMIC_CHUNK_EMA` | EMA weight λ | 0.3 |
| `DYNAMIC_CHUNK_ALPHA_MIN_PREFILL` | min prefill tokens to update α | 256 |

## step() pseudocode

```python
def _step_hslo(self, decode_depth, last_tokens):
    now = time.monotonic()
    dt = (now - self._last_t) * 1000.0 if self._last_t else None
    self._last_t = now
    if dt is not None and last_tokens is not None:
        prefill_last = max(0, last_tokens - self._prev_decode_depth)
        if prefill_last == 0:
            self.decode_baseline = self._ema(self.decode_baseline, dt)
        elif prefill_last >= self.alpha_min_prefill:
            a = max(dt - (self.decode_baseline or 0.0), 1e-3) / prefill_last
            self.alpha = self._ema(self.alpha, a)
    self._prev_decode_depth = decode_depth

    if decode_depth == 0:
        budget = self.max                                   # nothing to protect
    elif self.decode_baseline is None or self.alpha is None:
        budget = self.start                                 # warmup
    else:
        headroom = self.slo_ms - self.decode_baseline
        budget = self.min if headroom <= 0 else \
                 int(min(self.max, max(self.min, headroom / self.alpha)))
    self.chunk = budget
    self._trace(decode_depth, float(budget))
    return budget
```

## Testing / evaluation

Add arm `bhslo` (mode=hslo, pad-seed 1001) via a new `orchestrate_longprompt_hslo.sh` mirroring
the depth/ff orchestrators, then analyze with
`BUDGETS="16384 2048 512 ours ffv1 ffv2 depth hslo"`. Success = TBT-p99 near chunk (≤ ~450 ms,
i.e. −80%+) **and** TBT-max bounded (no 6 s leak, unlike slocvar/depth) **and** TTFT tax
materially below static-512's +20% (ideally single-digit %). Controller trace should dwell in the
interior (meaningful time at 1024–2048), not rail floor/ceiling — that's the discriminating
signature versus all four prior arms.

## Out of scope (YAGNI)

No 2-variable regression, no CVaR, no multi-SLO. One decode EMA + one α EMA + clamp. If α proves
noisy we revisit, but not pre-emptively.
