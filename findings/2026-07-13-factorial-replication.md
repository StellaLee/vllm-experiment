# Mechanism factorial — REPLICATION (3 trials): the herd win is chunking, and it holds

**Date:** 2026-07-13
**Motivation:** The original mechanism factorial (findings/2026-07-10-mechanism-factorial.md)
was **single-trial**, the paper's headline empirical claim, and the reviewer punch-list's
"single-trial fatal irony" (R1). The Cs² crossover just FAILED to replicate
(findings/2026-07-13-cs2-crossover.md), so we must check whether the herd attribution is
also noise. Prediction: unlike the small open-loop Cs² effect, the herd win is a LARGE
effect in the reproducible closed-loop regime → should hold.

## Setup
Identical to the 2026-07-10 factorial, HERD arms only, **3 trials** (seeds r1/r2/r3): single
RTX 4090, Qwen2.5-Coder-7B, vLLM 0.23.0, real ShareGPT multi-turn (prefix caching ON, NO
padding → warmth exists), synchronized closed-loop c=15, 120 convs × 4 turns,
max_tokens=128, max-num-batched-tokens=2048, depth-mode chunk. Arms {baseline, reorder,
chunk, combined}. Metric: per-turn TTFT p95, % vs baseline, **paired within each trial**,
reported mean ± std across trials (`scripts/analyze_mfact_repl.py`).

## Result — % vs baseline (mean ± std across 3 trials, wins/3)
| Turn | reorder | chunk | combined |
|---|---|---|---|
| T1 | +3.1 ±13.4 (1/3) | −7.2 ±10.6 (2/3) | +0.6 ±19.3 (2/3) |
| T2 | +1.4 ±5.5 (2/3) | **−14.3 ±8.5 (3/3)** | −5.6 ±6.3 (2/3) |
| T3 | −3.3 ±4.9 (2/3) | **−34.9 ±1.6 (3/3)** | **−32.7 ±9.5 (3/3)** |
| T4 | +6.7 ±17.8 (1/3) | −8.2 ±15.9 (2/3) | −12.0 ±19.5 (2/3) |
per-trial chunk T3: [−35 / −33 / −37].

## Conclusion
- **The headline holds, tightly.** The herd win is CHUNKING: chunk T3 = **−34.9% ± 1.6, 3/3**
  (an exceptionally tight large effect), and chunk T2 = −14.3% ± 8.5, 3/3. Reordering does
  NOT drive it: reorder is neutral at every turn (straddles zero, small ±). This confirms
  the §4 re-attribution (chunk, not reorder) and **discharges R1**.
- **Confirms the Cs²-noise contrast:** the large closed-loop effect replicated cleanly,
  exactly where the small open-loop Cs² effect did not — consistent with "large effects
  survive noise, small ones don't," not with "single trials here are all noise."
- **Honest refinement — soften the finer numbers.** T3 is the robust win; T2 also solid.
  But T1 and especially **T4 are noisy** (chunk T4 −8 ±16, was −26% single-trial; one trial
  went +8). The paper should lead on T3/T2 and drop/soften the exact T4 −26% and T1 figures.
- Combined ≈ chunk (T3 −33% vs −35%), reconfirming reorder adds nothing on top of chunk.

Raw logs: `logs/2026-07-13-mfact-{baseline,reorder,chunk,combined}_herd_t{r1,r2,r3}.jsonl`.
Analyzer: `scripts/analyze_mfact_repl.py`.
