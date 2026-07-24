# Whale-fraction x whale-size grid: the decode-protection win generalizes, but p99 flips sign at low whale frequency

**Date:** 2026-07-24
**Box:** 8×4090, `183.147.142.123`, vLLM 0.23.0 (V1), Qwen2.5-Coder-14B-Instruct, TP=2
**Workload:** `orchestrate_whale_grid.sh` — budget mechanism (mono=16384 vs chunk=2048, the
§5.6 lever), concurrency=20 (established safe point), single-turn, max-tokens=256, NCONV=200,
pad-seed=1001. Grid: whale size in {8k tok (~24-28k chars), 14k tok (~44-50k chars, the
already-validated range)} x whale fraction in {5%, 15%, 30%} = 6 cells x 2 budgets = 12 runs.
This maps the genuine term's frontier, flagged as the natural next step in
`2026-07-21-longprompt-tbt-win.md`'s own writeup and never executed until now.

**Headline: the decode-protection win (TTFT, TBT-max) generalizes robustly across the whole
tested frontier — but TBT-p99 flips sign at low whale frequency, for a mechanistically
explainable reason.**

## Full results

| size | frac | budget | n | nwhale | TTFT mean | TBT p99 | TBT max |
|---|---|---|---|---|---|---|---|
| 8k | 05 | 16384 | 200 | 10 | 795 | 191.2 | 2718.2 |
| 8k | 05 | 2048 | 200 | 10 | 593 | 362.4 | 752.0 |
| 8k | 15 | 16384 | 200 | 36 | 1474 | 1501.1 | 4877.9 |
| 8k | 15 | 2048 | 200 | 36 | 1064 | 389.6 | 905.2 |
| 8k | 30 | 16384 | 200 | 63 | 3035 | 1615.9 | 2982.9 |
| 8k | 30 | 2048 | 200 | 63 | 2104 | 396.0 | 1142.3 |
| 14k | 05 | 16384 | 199 | 9 | 1212 | 199.1 | 5948.1 |
| 14k | 05 | 2048 | 199 | 9 | 928 | 410.8 | 1175.9 |
| 14k | 15 | 16384 | 196 | 32 | 5647 | 2463.9 | 5882.9 |
| 14k | 15 | 2048 | 196 | 32 | 4955 | 438.1 | 1190.6 |
| 14k | 30 | 16384 | 190\* | 53 | 12168 | 2814.4 | 5060.3 |
| 14k | 30 | 2048 | 190\* | 53 | 10015 | 442.8 | 903.6 |

\* 10/200 requests (5%) hit HTTP 400 context-overflow errors in this cell, identically across
both budget arms (an admission-time check independent of scheduling policy). Likely because
real ShareGPT text occasionally tokenizes denser than the filler's calibrated 3.235
chars/token ratio (e.g., code snippets), pushing the largest whale+real-content combinations
over the limit despite the char-based guard. Smaller and much less severe than the earlier
27-29/200 bug; this cell's numbers should be read with that caveat rather than discarded.

## Delta: chunk=2048 vs. mono=16384

| size | frac | dTTFT% | dTBT-p99% | dTBT-max% |
|---|---|---|---|---|
| 8k | 05 | −25.5 | **+89.5** | −72.3 |
| 8k | 15 | −27.8 | −74.0 | −81.4 |
| 8k | 30 | −30.7 | −75.5 | −61.7 |
| 14k | 05 | −23.4 | **+106.3** | −80.2 |
| 14k | 15 | −12.3 | −82.2 | −79.8 |
| 14k | 30 | −17.7 | −84.3 | −82.1 |

## Interpretation

**TTFT and TBT-max are robust across the whole grid.** Chunk=2048 improves TTFT (−12% to
−31%) and TBT-max (−62% to −82%) at every single cell, regardless of whale size or fraction.
The core decode-protection mechanism from `longprompt-tbt-win` is not a fragile, single-point
result — it holds across a real frontier of whale characteristics.

**TBT-p99 flips sign at low whale frequency (5%), for both sizes, and this is mechanistically
explainable rather than noise.** P99 requires 1% of *all* pooled output tokens to exceed the
cutoff. At wf=5%, mono has only 9-10 whale events across 200 requests — too infrequent for
their (catastrophic) freezes to dominate the top 1% of the pooled token distribution, so
mono's own p99 is set mostly by ordinary decode noise. Chunk=2048 trades those rare,
catastrophic freezes for many more frequent, moderately-elevated gaps (each whale now takes
~6-7 chunked iterations instead of 1) — frequent enough, even at low whale density, to
actually *raise* the p99 statistic, even while cutting the max dramatically (chunking never
stops protecting the single worst case). At wf=15/30%, there are enough mono-driven
catastrophic events that they dominate the p99 cutoff directly, and chunking's improvement
shows through cleanly in both directions.

**This means "does chunking help p99 TBT" is conditional on whale *frequency*, not just
size** — a frontier boundary the original single-point result (16% whale fraction) never
surfaced, because it happened to sit comfortably in the "frequent enough" regime. The max
statistic doesn't have this problem (it only cares about the single worst event, not a
population threshold), so "chunking bounds the worst case" is the more universally robust
claim; "chunking improves the tail statistic you'd report in a paper (p99)" is the
frequency-conditional one.

**Whale size and fraction compound superlinearly on pooled TTFT.** Mono's pooled TTFT at
wf=30/14k reaches 12,168ms — by far the highest load point tested, and the one cell that lost
requests to context overflow. This cell is likely approaching an overloaded-like regime
itself; read its exact numbers cautiously, similar to how conc=40 was flagged in the
whale/threshold/concurrency work.

## Caveats

- Single trial per cell (12 runs total, no repeats).
- The 14k/wf=30% cell lost 5% of requests to context overflow — noted above, treat that row's
  exact numbers as approximate.
- Only two budget arms tested (mono, chunk=2048) — chunk=512 was not included in this grid to
  keep the run tractable; the interior-optimum comparison (2048 vs 512) is not re-mapped here.
- "nwhale" counts use a `pad_chars>=20000` cutoff (lower than the usual 40000, since the 8k
  size band's whales are only 24-28k chars) — correctly separates whale/short for both size
  bands given the short population's pad is bounded to [100,8000] chars throughout.

## Next steps

1. The p99-flips-at-low-frequency finding is novel and worth its own dedicated write-up in the
   paper (likely as a qualification to §5.6's TBT-p99 claim, alongside the already-robust
   TBT-max claim).
2. Repeat-trial this grid, or at minimum the two sign-flip cells (8k/05, 14k/05), to confirm
   the p99 reversal is stable and not itself a single-run artifact.
3. Consider adding chunk=512 to complete the interior-optimum comparison across the grid.
4. Feeds `docs/2026-07-24-mlsys-main-track-plan.md` Phase 2 — this closes out the
   whale-fraction x whale-size grid sub-task (3/3 Phase 2 pieces now done).
