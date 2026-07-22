# Open-loop Poisson: the whale decode-protection regime reproduces (and amplifies)

**Date:** 2026-07-22
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2, GPUs 0,1
**Builds on:** [2026-07-21-longprompt-tbt-win](2026-07-21-longprompt-tbt-win.md) (the closed-loop,
concurrency-20 version of this experiment).
**Headline:** Driving the identical bimodal-whale workload **open-loop (Poisson arrivals)** instead of
closed-loop concurrency-20, chunk still cuts mono's decode-freeze tail by **−84% (2048) / −95% (512)**
on P99 TBT. The effect is if anything **amplified**: mono's multi-second whale iteration stalls the
scheduler, so arrivals queue behind it and the decode batch mono freezes grows *larger* than in
closed-loop (realized concurrency ~37 vs 20). This answers "was the regime a closed-loop artifact?" —
no.

---

## TL;DR

- The whale-freeze regime is **arrival-pattern-independent**: it reproduces under Poisson arrivals with
  the same magnitude (−84%/−95% P99 TBT) as closed-loop.
- Open-loop **amplifies** it via queueing: mono's frozen scheduler backs up arrivals, so mean in-flight
  concurrency ran **37 (max 67)** at a rate that Little's law would put at ~20 if latency matched the
  reference — a bigger decode batch to freeze.
- The **TTFT tradeoff shifts**: closed-loop, chunk-2048 was a *free* win (−11% TTFT); open-loop it
  costs a little TTFT (+4.5%) because open-loop TTFT is dominated by queue wait, which all arms pay.
  The tail win is unchanged; the "2048 also improves TTFT" claim is closed-loop-specific.

---

## Setup

- **Workload (identical to the 07-22 hslo line):** single-turn, real ShareGPT first-turns. Bimodal
  padding: ~84% short (base pad ~800 chars) + **~16% whales** with pad uniform in [44k, 50k] chars
  ≈ **~12k tokens**. Context guard caps total prompt at 50k chars. Realized: n=196, prompt words
  p50/p95/max = 210 / 9269 / 9819; **whales = 32 (16%)**. `--pad-seed 1001`.
- **Driver — OPEN-LOOP Poisson:** `--rate 1.232 conv/s`, arrivals as an exponential inter-arrival
  process (`--concurrency` ignored). The rate is **derived, not guessed**: it is the realized
  throughput of the 07-21 closed-loop conc-20 chunk-2048 reference
  (`logs/2026-07-21-longp-b2048-t1.jsonl`: throughput 1.232 conv/s, realized concurrency mean 19.6 /
  max 20). By Little's law this anchors the open-loop load to the population that produced the original
  signal.
- **Arms (isolated sequential, each alone on GPUs 0,1):** mono=16384, chunk=2048, chunk=512,
  all `DYNAMIC_CHUNK=0`, `--max-num-seqs 48 --max-model-len 16384 --gpu-memory-utilization 0.90`,
  `--max-tokens 256`, `--num-convs 200`.
- **Metric:** pooled **P99 / max TBT** (per-token ITL, Sarathi's metric) + TTFT + per-request-mean TPOT.
- **Validity gate (the load-bearing check for open-loop):** the whale-freeze only shows if a live
  decode batch exists when a whale prefills. The analyzer reports realized concurrency per arm; a clean
  mono tail would mean "rate too low, raise it," not a negative result. **Gate passed** (below).
- **Arrival rate (Derived)**: λ = 1.232 conv/s

## Results

```
 budget |    n TTFTmean    p95  dTTFT% | TPOTp50 TPOTp95 | TBTp50  TBTp99  TBTmax |    dP99    dMax  [arm]
16384ol |  195    18100  31826    +0.0 |    66.0   151.3 |   32.4  2738.7  5582.0 |    +0.0    +0.0  [mono]
 2048ol |  195    18916  35927    +4.5 |    75.9   181.2 |   35.3   442.4  1683.2 |   -83.8   -69.8  [chunk]
  512ol |  195    21116  36779   +16.7 |    71.5   115.8 |   36.7   141.3   402.0 |   -94.8   -92.8  [chunk]
```
(TTFT/TBT in ms. 195/195 completed each arm.)

**Validity gate — realized concurrency:**

| arm | n | realized conc mean | max | throughput |
|---|---|---|---|---|
| mono 16384 | 196 | **37.4** | 67 | 1.253/s |
| chunk 2048 | 196 | 37.3 | 66 | 1.198/s |
| chunk 512 | 196 | 37.6 | 63 | 1.155/s |

Mono's P99 TBT is 2739ms (≫ its ~32ms decode baseline) — the freeze is present, the gate passes.

## Interpretation

**The regime is arrival-pattern-independent.** In continuous batching every decoder advances one token
per scheduler iteration, so a whale that prefills in one giant mono iteration freezes every co-resident
decoder for that iteration — whether the whale arrived on a closed-loop heartbeat or a Poisson draw is
irrelevant. Chunk-2048 slices the whale into ~6 bounded iterations (P99 TBT 442ms), chunk-512 into ~24
(141ms). Same mechanism, same magnitude as closed-loop (−84%/−95%).

**Open-loop amplifies the effect through queueing.** At rate 1.232 conv/s, Little's law predicts
in-flight ≈ 20 *if* mean latency matched the reference. Instead realized concurrency ran **~37 (max
67)** — because mono's multi-second whale iterations stall the scheduler, arrivals pile up behind them,
mean latency inflates, and the in-flight population balloons. So open-loop doesn't just preserve the
regime, it puts a *bigger* decode batch under the whale's freeze. (Consistent across arms: even chunk's
lighter-tailed arms sit at ~37 concurrency, since the rate is fixed and the whales' KV load dominates
latency.)

**The TTFT tradeoff is different from closed-loop — and the difference is instructive.** Closed-loop,
chunk-2048 *improved* TTFT (−11%): with a bounded population, slicing the whale let short requests
interleave sooner. Open-loop, TTFT is dominated by **queue wait** (absolute TTFT ~18s on every arm,
because the load runs hot at concurrency 37–67), which all arms pay roughly equally; chunk-2048 then
shows a small TTFT *cost* (+4.5%) and chunk-512 a larger one (+16.7%). The tail win is unchanged, but
the "2048 is a free lunch on TTFT too" claim from 07-21 is **closed-loop-specific**. Under any
interactive TBT SLO ("no >500ms freeze"), mono still FAILS (2739ms) and chunk-2048 PASSES (442ms) —
the win holds; it is just no longer free on TTFT.

## Caveats

- **Single trial**, n=32 whales, one whale size/fraction (directionally overwhelming: −84–95% ≫ noise).
- **Ran hot.** The derived rate landed the system in a **queue-heavy** regime (concurrency 37–67, well
  above the reference's 20; absolute TTFT ~18s). Two reasons: (1) open-loop queueing amplification
  above; (2) the rate anchor came from the 07-21 reference, which used **larger** whales (48–60k chars,
  context cap 62k) than this run (44–50k, cap 50k) — a heavier reference latency → the anchored rate is
  a touch aggressive for the lighter whales. The **regime conclusion is robust** (the freeze and the
  −84/−95% cut are unambiguous), but the absolute TTFT numbers reflect a near-saturation operating
  point, not a chosen SLO. The natural follow-up is a **rate sweep** to map goodput (max QPS under a
  P99-TBT SLO) — Sarathi's headline framing — rather than a single hot point.
- `whale_max == max_prompt_chars` (both 50k) clips the top of the whale range; kept identical to the
  07-22 hslo workload for cross-experiment comparability rather than widened here.

## Artifacts

- Orchestrator: `orchestrate_longprompt_openloop.sh` (open-loop `--rate`, rate derived from the 07-21
  reference via `scripts/replay_timing.realized_throughput`, isolated sequential, validity-gate footer).
- Timing helpers: `scripts/replay_timing.py` (`realized_throughput`, `realized_concurrency` for the
  gate; unit-tested in `tests/test_replay_timing.py`).
- Analyzer: `scripts/analyze_longprompt.py` (core P99/max TBT + TTFT deltas vs mono) + the concurrency
  validity footer in the orchestrator.
- Data: `logs/2026-07-22-longp-b{16384,2048,512}ol-t1.jsonl`, analysis `logs/longpol_ANALYSIS.txt`.

## Next

- **Rate sweep / goodput frontier** — sweep `--rate` from sub-saturation upward, report max QPS under a
  P99-TBT SLO for each budget. Turns this single hot point into the goodput headline and separates the
  budgets on the frontier rather than at one over-driven load.
- The non-stationary (dynamic-chunking) experiment is tracked separately; a de-risk pilot on this same
  box found `decode_baseline` is a small fraction of the 400ms SLO across the attainable concurrency
  range, which reshapes that experiment's design (see the SDD progress ledger / follow-up finding).
