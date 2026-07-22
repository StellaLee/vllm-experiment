# Dynamic chunking: the SLO-headroom feedforward controller (hslo)

**Date:** 2026-07-22
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** identical to [2026-07-21-longprompt-tbt-win](2026-07-21-longprompt-tbt-win.md) — bimodal
whales (~16% × ~12k tokens + short majority), single-turn, concurrency 20, isolated sequential arms,
pad-seed 1001 (whale positions **paired** across every arm). Metric: pooled **P99 / max TBT**.
**Headline:** A launch-time-configured controller (`CHUNK_MODE=hslo`), given only **offline-computable
knobs**, autonomously operates at the interior 2048 chunk that the 07-21 sweep found by hand — matching
the oracle static-2048's decode protection (P99 tie, **max better**), at a TTFT cost on stationary load.

---

## TL;DR

- **Four naive controllers all fail by railing to a corner** (≈mono or ≈static-512); none finds the
  interior. The failure is structural, not tuning.
- **hslo works** because it sizes the prefill budget from **measured decode _cost_** (not a decoder
  count) plus a **feedforward** prefill term (known before the whale is scheduled, so it's not one
  iteration late).
- The controller's SLO is a **per-iteration time cap ≈ a TBT tolerance**, and it has a clean
  **offline heuristic: `SLO_MS = wall-time of one isolated 2048-token prefill iteration`** (≈400ms
  here) — i.e. adopt vLLM's own default chunk as the latency contract.
- Two bugs found and fixed by trace diagnosis: **(1) warmup-at-ceiling** first-whale leak → start at
  the floor; **(2) α-underestimation railing** the budget to 16384 under load → **floor α at the
  offline hardware cost**. After both, TBT-max collapses 6021 → **1094** and the interior locks at
  **median 1840 / p90 2068**.

---

## The four controllers that fail (and why)

All start high (16384) and must drop when a whale threatens decoders. Paired whales, same workload.

| controller | basis | outcome | TBT-p99 | TBT-max | dwell |
|---|---|---|---|---|---|
| depth + hysteresis | decode **count** vs target | floor + max leak | 385 (−84%) | 6009 | floor 80% |
| slocvar | windowed CVaR of past latency | floor + max leak | 136 (−94%) | 5992 | floor 71% |
| feedforward-v1 | τ = dt / **granted** budget | ≈ mono | 1898 (−23%) | 5767 | **ceil 93%** |
| feedforward-v2 | τ = dt / **actual** tokens | ≈ static-512 | 132 (−95%) | 391 | **floor 100%** |

Two structural faults:
1. **Count ≠ cost.** decode-depth is blind to *how expensive* the current decode batch is (a shallow
   but slow batch has no headroom, yet depth says "grow"). A cost-aware controller must measure time.
2. **Reactive ≠ in-time.** A whale prefills in **one** iteration, so any controller that sets the
   budget from the *previous* step's latency is a step behind — it learns the whale only after the 6s
   freeze already happened. The feedback loop cannot close inside one iteration.

`ffv2`/`slocvar`/`depth` "win" p99 only by collapsing to the floor (= static-512 in disguise), and the
ones that still dither (slocvar, depth) leak the 6s max because they were mid-transition when the first
whale hit. `ffv1` stays timid at the ceiling ≈ mono. **No controller finds the 2048 knee.**

## The design that works: SLO-headroom feedforward

Per iteration, model `iter_time ≈ decode_baseline + α·prefill_tokens` and solve for the budget:

```
budget = clamp( (SLO − decode_baseline) / α_eff , floor, ceiling )
α_eff  = max( α_online , α_hw )          # α_hw = offline hardware floor (see fix #2)
```

- **decode_baseline** — a *measured* decode-only iteration time (answers count-vs-cost). Estimated by
  a **segregated estimator**: updated only on pure-decode steps (`prefill_last == 0`). α is updated
  only on prefill-heavy steps (`prefill_last ≥ 256`) as `(dt − decode_baseline)/prefill_last`. The two
  costs are **never blended** — a single `EMA(dt/total_tokens)` is dominated by cheap decode steps and
  overestimates α ~4× → rails to floor (that's the slocvar/ffv2/depth failure, reproduced on paper).
- **prefill term is feedforward** — the budget *is* the prefill token count, chosen before scheduling,
  so the whale is capped on arrival, not one iteration late.
- `prefill_last = last_tokens − prev_decode_depth` — derived, no new plumbing (each decoder advances
  exactly one token/iteration).
- **Zero-decode exception:** `decode_depth == 0 → ceiling` (a presence check, not a cost proxy) — an
  idle whale prefills at full speed for best TTFT when nothing is at risk.

## The SLO heuristic (offline-computable, since SLO_MS is a launch-time env)

The controller's `SLO_MS` is a **per-iteration wall-time cap**; in continuous batching per-iteration
time ≈ the inter-token latency (TBT) felt by every decoder → **`SLO_MS` is a TBT tolerance**. It must
be set before the server starts, so it can't be read from live traffic. The anchor is vLLM's default
`max-num-batched-tokens = 2048`, an implicit statement that a 2048-token iteration is acceptable:

> **`SLO_MS` = wall-time of one isolated 2048-token prefill iteration** (≈**400ms** here).

Because `SLO = α·2048 = (t₂₀₄₈/2048)·2048 = t₂₀₄₈`, the micro-benchmark *is* the answer — no α
indirection, and it self-uses the correct marginal cost at the 2048 operating point. Under this SLO the
controller reproduces 2048 at reference decode load and **tightens below 2048 as decode load grows**
(better protection) — the vLLM default becomes an adaptive operating point.

**SLO sweep confirms the mapping** (median chunk held, decoders present):

| SLO_MS | 50 | 200 | 400 | 600 | 1200 |
|---|---|---|---|---|---|
| median chunk | floor | 897 | **1840** | 3108 | 7344 |
| dwell | floor 97% | interior 97% | interior 96% | interior 71% | ceil 65% |

Monotonic, with 1000+ mid-run adjustments per run — hslo **traces the SLO→chunk frontier** that the
four corner-railing controllers never could. (Note the effective marginal α ≈ 0.19 ms/tok, lower than
the 0.37 read off the full 16k mono iteration; the 2048-iteration micro-benchmark captures the right
one.)

## Two bugs, found by trace diagnosis

**Fix #1 — warmup-at-floor.** With warmup at `start=16384`, the first whale (before α is learned)
prefilled whole → one 6s freeze in every arm. Starting at the floor (512) caps the first whale while α
learns: **TBT-max 6021 → 2949**.

**Fix #2 — α hardware floor.** Residual 2949ms leak was **not** the idle exception: the trace showed
**359/361 ceiling steps had `decode_depth>0` (mean 13.2 decoders)**. Cause: on noisy small-prefill
steps `dt−decode_baseline` is tiny → α underestimates (~8×) → `(SLO−db)/α` rails to 16384 → a whale
prefills whole while decoders are frozen. Fix: `α_eff = max(α_online, α_hw)` with `α_hw = 0.18 ms/tok`
(offline profile). This caps budget at `(SLO−db)/α_hw ≈ 2048` and blocks only the *downward* α
excursion. **TBT-max 2949 → 1094; ceiling steps 361 → 3, all now legitimately idle (`depth==0`).**

## Result — hslo vs the oracle static-2048

| arm | TTFT Δ | TBT-p99 | ΔP99 | TBT-max | ΔMax | interior median |
|---|---|---|---|---|---|---|
| static-2048 (oracle) | **−11.2%** | 437 | −82% | 1229 | −79% | fixed 2048 |
| static-512 | +20.6% | 137 | −94% | 392 | −93% | fixed 512 |
| hslo@400 warmup=ceil | +11.0% | 525 | −79% | 2949 | −51% | 1623 (ceil 15%) |
| **hslo@400 +wf +α-floor** | **+11.1%** | **438** | **−82%** | **1094** | **−82%** | **1840 (p90 2068)** |

- **P99: tie** with the oracle (438 vs 437). **Max: hslo wins** (1094 vs 1229). **TTFT: hslo loses**
  (+11.1% vs −11.2%), ~22-point gap.
- hslo reaches this with **only offline-computable knobs** (SLO=α·2048, α_hw from the same profile,
  warmup=floor) — no per-workload sweep. The controller autonomously *finds* the operating point that
  07-21 discovered by hand.

## Honest conclusions

1. **On a stationary workload, a well-tuned static budget is not beaten.** static-2048 holds exactly
   2048 and wins TTFT; hslo averages 1840 and warms from the floor, so it throttles prefill slightly
   more (+11% TTFT). hslo *matches the tail protection*, it doesn't dominate. Expected — a fixed load
   has a fixed optimum.
2. **The SLO is a _budget_ bound, not a hard _latency_ bound.** Max is 1094ms, not ≤400ms, because
   `α_hw=0.18` is a *lower* bound on true α; when real α spikes under decode contention (~0.5), a
   2048-chunk iteration costs ~1094ms. Tightening max toward 400 needs `α_hw≈0.5` → budget≈744 → worse
   TTFT. `α_hw` trades max-tightness against TTFT; 0.18 places hslo in static-2048's regime.
3. **The dynamic controller's real value is a _moving_ optimum.** Matching an oracle on stationary load
   is a sanity check; the payoff is **non-stationary load** (time-varying decode pressure / clustered
   whale bursts) where no single static budget is right, and hslo tracks it up and down. That is the
   next experiment and the honest case for the paper.

## Artifacts

- Controller: installed into `ChunkSizeController` via `scripts/hotpatch_hslo.py` (mode dispatch +
  `_step_hslo`) and `scripts/hotpatch_hslo_alphafloor.py` (α floor; idempotent, targeted). Env:
  `CHUNK_MODE=hslo`, `DYNAMIC_CHUNK_SLO_MS`, `DYNAMIC_CHUNK_MIN` (floor), `DYNAMIC_CHUNK_START`
  (warmup=floor), `DYNAMIC_CHUNK_ALPHA_MIN` (α_hw, ms/tok), `DYNAMIC_CHUNK_ALPHA_MIN_PREFILL`.
- Orchestrators: `orchestrate_longprompt_hslo.sh` (single @50), `_hslo_sweep.sh` (SLO 200/600/1200),
  `_hslo_wf.sh` (warmup-floor), `_hslo_af.sh` (α-floor, the final arm). Prior controllers:
  `_dyn.sh` (slocvar), `_ff.sh` (ffv1/ffv2), `_depth.sh`.
- Analyzer: `scripts/analyze_longprompt.py` (adds controller dwell/transition trace for non-numeric
  arms). Design spec: `docs/superpowers/specs/2026-07-22-slo-headroom-feedforward-design.md`.
- Data: `logs/2026-07-22-longp-bhslo*-t1.jsonl` + `-chunktrace.csv`.

## Next steps

1. **Non-stationary load** — alternating low/high concurrency phases (or time-clustered whale bursts)
   so the optimal budget moves; show hslo tracks it while static-2048 is wrong half the time. This is
   the experiment that converts hslo's +11% stationary TTFT cost into a win.
2. **Replicate** the α-floor arm 2–3 trials (single trial, n=32 whales) and sweep `α_hw` to map the
   max ↔ TTFT tradeoff.
3. **SLO-guarantee framing** — decide whether to claim a *budget* bound (honest, holds now) or invest
   in a per-iteration *latency* bound (needs α_hw = measured worst-case α, costs TTFT).
